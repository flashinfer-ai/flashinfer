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
cake_fp4_allreduce_tp4_twoshot_bf16_bv1_csp_b1r_rst(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, uint8_t* __restrict__ ts_wire, uint8_t* __restrict__ ts_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap)
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
            int g32 = (int)gridDim.x;
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
                int rs_wo_1 = (int)rs_w;
                int rs_ro_1 = (int)rs_r;
                int capo_1 = (int)capo64;
                int ix_1 = (int)tix64;
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
                int pp_1 = lane_1 + 32 * (3 * h);
                int pp_9 = lane_1 + 32 * (3 * h + 1);
                int pp_10_1 = lane_1 + 32 * (3 * h + 2);
                uint32_t _fp4_wire_values_0[48];
                uint32_t _fp4_wire_scales_0[12];
                uint16_t _wire_h16_143_0;
                uint16_t _wire_h16_143_1;
                uint16_t _wire_h16_143_2;
                uint16_t _wire_h16_143_3;
                uint16_t _wire_h16_143_4;
                uint16_t _wire_h16_143_5;
                uint16_t _wire_h16_143_6;
                uint16_t _wire_h16_143_7;
                uint16_t _wire_h16_143_8;
                uint16_t _wire_h16_143_9;
                uint16_t _wire_h16_143_10;
                uint16_t _wire_h16_143_11;
                uint32_t _wire_vpend_143 = 4095u;
                uint32_t _wire_spend_143 = 4095u;
                do {
                    if (_wire_vpend_143 & 1u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 2u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 4u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[8]), "=r"(_fp4_wire_values_0[9]), "=r"(_fp4_wire_values_0[10]), "=r"(_fp4_wire_values_0[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 8u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[12]), "=r"(_fp4_wire_values_0[13]), "=r"(_fp4_wire_values_0[14]), "=r"(_fp4_wire_values_0[15]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 16u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[16]), "=r"(_fp4_wire_values_0[17]), "=r"(_fp4_wire_values_0[18]), "=r"(_fp4_wire_values_0[19]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_9)))) : "memory");
                    }
                    if (_wire_vpend_143 & 32u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[20]), "=r"(_fp4_wire_values_0[21]), "=r"(_fp4_wire_values_0[22]), "=r"(_fp4_wire_values_0[23]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_9)))) : "memory");
                    }
                    if (_wire_vpend_143 & 64u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[24]), "=r"(_fp4_wire_values_0[25]), "=r"(_fp4_wire_values_0[26]), "=r"(_fp4_wire_values_0[27]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp_9)))) : "memory");
                    }
                    if (_wire_vpend_143 & 128u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[28]), "=r"(_fp4_wire_values_0[29]), "=r"(_fp4_wire_values_0[30]), "=r"(_fp4_wire_values_0[31]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp_9)))) : "memory");
                    }
                    if (_wire_vpend_143 & 256u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[32]), "=r"(_fp4_wire_values_0[33]), "=r"(_fp4_wire_values_0[34]), "=r"(_fp4_wire_values_0[35]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v + (int)(16 * pp_10_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 512u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[36]), "=r"(_fp4_wire_values_0[37]), "=r"(_fp4_wire_values_0[38]), "=r"(_fp4_wire_values_0[39]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_3 + (int)(16 * pp_10_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 1024u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[40]), "=r"(_fp4_wire_values_0[41]), "=r"(_fp4_wire_values_0[42]), "=r"(_fp4_wire_values_0[43]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_5 + (int)(16 * pp_10_1)))) : "memory");
                    }
                    if (_wire_vpend_143 & 2048u) {
                        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[44]), "=r"(_fp4_wire_values_0[45]), "=r"(_fp4_wire_values_0[46]), "=r"(_fp4_wire_values_0[47]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_7 + (int)(16 * pp_10_1)))) : "memory");
                    }
                    if (_wire_spend_143 & 1u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_1)))) : "memory");
                        _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_143_0;
                    }
                    if (_wire_spend_143 & 2u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp_1)))) : "memory");
                        _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_143_1;
                    }
                    if (_wire_spend_143 & 4u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_6 + (int)(2 * pp_1)))) : "memory");
                        _fp4_wire_scales_0[2] = (uint32_t)_wire_h16_143_2;
                    }
                    if (_wire_spend_143 & 8u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_3) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_8 + (int)(2 * pp_1)))) : "memory");
                        _fp4_wire_scales_0[3] = (uint32_t)_wire_h16_143_3;
                    }
                    if (_wire_spend_143 & 16u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_4) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_9)))) : "memory");
                        _fp4_wire_scales_0[4] = (uint32_t)_wire_h16_143_4;
                    }
                    if (_wire_spend_143 & 32u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_5) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp_9)))) : "memory");
                        _fp4_wire_scales_0[5] = (uint32_t)_wire_h16_143_5;
                    }
                    if (_wire_spend_143 & 64u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_6) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_6 + (int)(2 * pp_9)))) : "memory");
                        _fp4_wire_scales_0[6] = (uint32_t)_wire_h16_143_6;
                    }
                    if (_wire_spend_143 & 128u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_7) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_8 + (int)(2 * pp_9)))) : "memory");
                        _fp4_wire_scales_0[7] = (uint32_t)_wire_h16_143_7;
                    }
                    if (_wire_spend_143 & 256u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_8) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc + (int)(2 * pp_10_1)))) : "memory");
                        _fp4_wire_scales_0[8] = (uint32_t)_wire_h16_143_8;
                    }
                    if (_wire_spend_143 & 512u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_9) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_4 + (int)(2 * pp_10_1)))) : "memory");
                        _fp4_wire_scales_0[9] = (uint32_t)_wire_h16_143_9;
                    }
                    if (_wire_spend_143 & 1024u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_10) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_6 + (int)(2 * pp_10_1)))) : "memory");
                        _fp4_wire_scales_0[10] = (uint32_t)_wire_h16_143_10;
                    }
                    if (_wire_spend_143 & 2048u) {
                        asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_143_11) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_8 + (int)(2 * pp_10_1)))) : "memory");
                        _fp4_wire_scales_0[11] = (uint32_t)_wire_h16_143_11;
                    }
                    _wire_vpend_143 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_0[8] & ~((_fp4_wire_values_0[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[9] & ~((_fp4_wire_values_0[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[10] & ~((_fp4_wire_values_0[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[11] & ~((_fp4_wire_values_0[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2) | ((uint32_t)((((_fp4_wire_values_0[12] & ~((_fp4_wire_values_0[12] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[13] & ~((_fp4_wire_values_0[13] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[14] & ~((_fp4_wire_values_0[14] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[15] & ~((_fp4_wire_values_0[15] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 3) | ((uint32_t)((((_fp4_wire_values_0[16] & ~((_fp4_wire_values_0[16] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[17] & ~((_fp4_wire_values_0[17] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[18] & ~((_fp4_wire_values_0[18] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[19] & ~((_fp4_wire_values_0[19] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 4) | ((uint32_t)((((_fp4_wire_values_0[20] & ~((_fp4_wire_values_0[20] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[21] & ~((_fp4_wire_values_0[21] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[22] & ~((_fp4_wire_values_0[22] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[23] & ~((_fp4_wire_values_0[23] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 5) | ((uint32_t)((((_fp4_wire_values_0[24] & ~((_fp4_wire_values_0[24] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[25] & ~((_fp4_wire_values_0[25] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[26] & ~((_fp4_wire_values_0[26] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[27] & ~((_fp4_wire_values_0[27] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 6) | ((uint32_t)((((_fp4_wire_values_0[28] & ~((_fp4_wire_values_0[28] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[29] & ~((_fp4_wire_values_0[29] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[30] & ~((_fp4_wire_values_0[30] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[31] & ~((_fp4_wire_values_0[31] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 7) | ((uint32_t)((((_fp4_wire_values_0[32] & ~((_fp4_wire_values_0[32] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[33] & ~((_fp4_wire_values_0[33] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[34] & ~((_fp4_wire_values_0[34] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[35] & ~((_fp4_wire_values_0[35] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 8) | ((uint32_t)((((_fp4_wire_values_0[36] & ~((_fp4_wire_values_0[36] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[37] & ~((_fp4_wire_values_0[37] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[38] & ~((_fp4_wire_values_0[38] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[39] & ~((_fp4_wire_values_0[39] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 9) | ((uint32_t)((((_fp4_wire_values_0[40] & ~((_fp4_wire_values_0[40] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[41] & ~((_fp4_wire_values_0[41] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[42] & ~((_fp4_wire_values_0[42] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[43] & ~((_fp4_wire_values_0[43] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 10) | ((uint32_t)((((_fp4_wire_values_0[44] & ~((_fp4_wire_values_0[44] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[45] & ~((_fp4_wire_values_0[45] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[46] & ~((_fp4_wire_values_0[46] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[47] & ~((_fp4_wire_values_0[47] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 11);
                    _wire_spend_143 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[3] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[3] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 3) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[4] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[4] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 4) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[5] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[5] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 5) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[6] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[6] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 6) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[7] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[7] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 7) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[8] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[8] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 8) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[9] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[9] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 9) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[10] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[10] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 10) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[11] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[11] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 11);
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
                float2 a_11 = make_float2(acc[2], acc[3]);
                float2 q_12 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                float2 r_13 = fma_f32x2_rn_noftz(q_12, scale0, a_11);
                acc[2] = r_13.x;
                acc[3] = r_13.y;
                float2 a_14 = make_float2(acc[4], acc[5]);
                float2 q_15 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                float2 r_16 = fma_f32x2_rn_noftz(q_15, scale0, a_14);
                acc[4] = r_16.x;
                acc[5] = r_16.y;
                float2 a_17 = make_float2(acc[6], acc[7]);
                float2 q_18 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                float2 r_19 = fma_f32x2_rn_noftz(q_18, scale0, a_17);
                acc[6] = r_19.x;
                acc[7] = r_19.y;
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
                float2 a_20 = make_float2(acc[8], acc[9]);
                float2 q_21 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                float2 r_22 = fma_f32x2_rn_noftz(q_21, scale0, a_20);
                acc[8] = r_22.x;
                acc[9] = r_22.y;
                float2 a_23 = make_float2(acc[10], acc[11]);
                float2 q_24 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                float2 r_25 = fma_f32x2_rn_noftz(q_24, scale0, a_23);
                acc[10] = r_25.x;
                acc[11] = r_25.y;
                float2 a_26 = make_float2(acc[12], acc[13]);
                float2 q_27 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                float2 r_28 = fma_f32x2_rn_noftz(q_27, scale0, a_26);
                acc[12] = r_28.x;
                acc[13] = r_28.y;
                float2 a_29 = make_float2(acc[14], acc[15]);
                float2 q_30 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                float2 r_31 = fma_f32x2_rn_noftz(q_30, scale0, a_29);
                acc[14] = r_31.x;
                acc[15] = r_31.y;
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
                float2 a_32_1 = make_float2(acc[16], acc[17]);
                float2 q_33 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                float2 r_34 = fma_f32x2_rn_noftz(q_33, scale1, a_32_1);
                acc[16] = r_34.x;
                acc[17] = r_34.y;
                float2 a_35 = make_float2(acc[18], acc[19]);
                float2 q_36 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                float2 r_37 = fma_f32x2_rn_noftz(q_36, scale1, a_35);
                acc[18] = r_37.x;
                acc[19] = r_37.y;
                float2 a_38 = make_float2(acc[20], acc[21]);
                float2 q_39 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                float2 r_40 = fma_f32x2_rn_noftz(q_39, scale1, a_38);
                acc[20] = r_40.x;
                acc[21] = r_40.y;
                float2 a_41 = make_float2(acc[22], acc[23]);
                float2 q_42 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                float2 r_43 = fma_f32x2_rn_noftz(q_42, scale1, a_41);
                acc[22] = r_43.x;
                acc[23] = r_43.y;
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
                float2 a_44 = make_float2(acc[24], acc[25]);
                float2 q_45 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                float2 r_46 = fma_f32x2_rn_noftz(q_45, scale1, a_44);
                acc[24] = r_46.x;
                acc[25] = r_46.y;
                float2 a_47 = make_float2(acc[26], acc[27]);
                float2 q_48 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                float2 r_49 = fma_f32x2_rn_noftz(q_48, scale1, a_47);
                acc[26] = r_49.x;
                acc[27] = r_49.y;
                float2 a_50 = make_float2(acc[28], acc[29]);
                float2 q_51 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                float2 r_52 = fma_f32x2_rn_noftz(q_51, scale1, a_50);
                acc[28] = r_52.x;
                acc[29] = r_52.y;
                float2 a_53 = make_float2(acc[30], acc[31]);
                float2 q_54 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                float2 r_55 = fma_f32x2_rn_noftz(q_54, scale1, a_53);
                acc[30] = r_55.x;
                acc[31] = r_55.y;
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
                float2 scale0_56 = make_float2(_mul_2, _mul_2);
                float2 scale1_57 = make_float2(_mul_3, _mul_3);
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
                float2 a_58 = make_float2(acc[0], acc[1]);
                float2 q_59 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                float2 r_60 = fma_f32x2_rn_noftz(q_59, scale0_56, a_58);
                acc[0] = r_60.x;
                acc[1] = r_60.y;
                float2 a_61 = make_float2(acc[2], acc[3]);
                float2 q_62 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                float2 r_63 = fma_f32x2_rn_noftz(q_62, scale0_56, a_61);
                acc[2] = r_63.x;
                acc[3] = r_63.y;
                float2 a_64 = make_float2(acc[4], acc[5]);
                float2 q_65 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                float2 r_66 = fma_f32x2_rn_noftz(q_65, scale0_56, a_64);
                acc[4] = r_66.x;
                acc[5] = r_66.y;
                float2 a_67 = make_float2(acc[6], acc[7]);
                float2 q_68 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                float2 r_69 = fma_f32x2_rn_noftz(q_68, scale0_56, a_67);
                acc[6] = r_69.x;
                acc[7] = r_69.y;
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
                float2 a_70 = make_float2(acc[8], acc[9]);
                float2 q_71 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                float2 r_72 = fma_f32x2_rn_noftz(q_71, scale0_56, a_70);
                acc[8] = r_72.x;
                acc[9] = r_72.y;
                float2 a_73 = make_float2(acc[10], acc[11]);
                float2 q_74 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                float2 r_75 = fma_f32x2_rn_noftz(q_74, scale0_56, a_73);
                acc[10] = r_75.x;
                acc[11] = r_75.y;
                float2 a_76 = make_float2(acc[12], acc[13]);
                float2 q_77 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                float2 r_78 = fma_f32x2_rn_noftz(q_77, scale0_56, a_76);
                acc[12] = r_78.x;
                acc[13] = r_78.y;
                float2 a_79 = make_float2(acc[14], acc[15]);
                float2 q_80 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                float2 r_81 = fma_f32x2_rn_noftz(q_80, scale0_56, a_79);
                acc[14] = r_81.x;
                acc[15] = r_81.y;
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
                float2 a_82 = make_float2(acc[16], acc[17]);
                float2 q_83 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                float2 r_84 = fma_f32x2_rn_noftz(q_83, scale1_57, a_82);
                acc[16] = r_84.x;
                acc[17] = r_84.y;
                float2 a_85 = make_float2(acc[18], acc[19]);
                float2 q_86 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                float2 r_87 = fma_f32x2_rn_noftz(q_86, scale1_57, a_85);
                acc[18] = r_87.x;
                acc[19] = r_87.y;
                float2 a_88 = make_float2(acc[20], acc[21]);
                float2 q_89 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                float2 r_90 = fma_f32x2_rn_noftz(q_89, scale1_57, a_88);
                acc[20] = r_90.x;
                acc[21] = r_90.y;
                float2 a_91 = make_float2(acc[22], acc[23]);
                float2 q_92 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                float2 r_93 = fma_f32x2_rn_noftz(q_92, scale1_57, a_91);
                acc[22] = r_93.x;
                acc[23] = r_93.y;
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
                float2 a_94 = make_float2(acc[24], acc[25]);
                float2 q_95 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                float2 r_96 = fma_f32x2_rn_noftz(q_95, scale1_57, a_94);
                acc[24] = r_96.x;
                acc[25] = r_96.y;
                float2 a_97 = make_float2(acc[26], acc[27]);
                float2 q_98 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                float2 r_99 = fma_f32x2_rn_noftz(q_98, scale1_57, a_97);
                acc[26] = r_99.x;
                acc[27] = r_99.y;
                float2 a_100 = make_float2(acc[28], acc[29]);
                float2 q_101 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                float2 r_102 = fma_f32x2_rn_noftz(q_101, scale1_57, a_100);
                acc[28] = r_102.x;
                acc[29] = r_102.y;
                float2 a_103 = make_float2(acc[30], acc[31]);
                float2 q_104 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                float2 r_105 = fma_f32x2_rn_noftz(q_104, scale1_57, a_103);
                acc[30] = r_105.x;
                acc[31] = r_105.y;
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
                float2 scale0_106 = make_float2(_mul_4, _mul_4);
                float2 scale1_107 = make_float2(_mul_5, _mul_5);
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
                float2 a_108 = make_float2(acc[0], acc[1]);
                float2 q_109 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                float2 r_110 = fma_f32x2_rn_noftz(q_109, scale0_106, a_108);
                acc[0] = r_110.x;
                acc[1] = r_110.y;
                float2 a_111 = make_float2(acc[2], acc[3]);
                float2 q_112 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                float2 r_113 = fma_f32x2_rn_noftz(q_112, scale0_106, a_111);
                acc[2] = r_113.x;
                acc[3] = r_113.y;
                float2 a_114 = make_float2(acc[4], acc[5]);
                float2 q_115 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                float2 r_116 = fma_f32x2_rn_noftz(q_115, scale0_106, a_114);
                acc[4] = r_116.x;
                acc[5] = r_116.y;
                float2 a_117 = make_float2(acc[6], acc[7]);
                float2 q_118 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                float2 r_119 = fma_f32x2_rn_noftz(q_118, scale0_106, a_117);
                acc[6] = r_119.x;
                acc[7] = r_119.y;
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
                float2 a_120 = make_float2(acc[8], acc[9]);
                float2 q_121 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                float2 r_122 = fma_f32x2_rn_noftz(q_121, scale0_106, a_120);
                acc[8] = r_122.x;
                acc[9] = r_122.y;
                float2 a_123 = make_float2(acc[10], acc[11]);
                float2 q_124 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                float2 r_125 = fma_f32x2_rn_noftz(q_124, scale0_106, a_123);
                acc[10] = r_125.x;
                acc[11] = r_125.y;
                float2 a_126 = make_float2(acc[12], acc[13]);
                float2 q_127 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                float2 r_128 = fma_f32x2_rn_noftz(q_127, scale0_106, a_126);
                acc[12] = r_128.x;
                acc[13] = r_128.y;
                float2 a_129 = make_float2(acc[14], acc[15]);
                float2 q_130 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                float2 r_131 = fma_f32x2_rn_noftz(q_130, scale0_106, a_129);
                acc[14] = r_131.x;
                acc[15] = r_131.y;
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
                float2 a_132 = make_float2(acc[16], acc[17]);
                float2 q_133 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                float2 r_134 = fma_f32x2_rn_noftz(q_133, scale1_107, a_132);
                acc[16] = r_134.x;
                acc[17] = r_134.y;
                float2 a_135 = make_float2(acc[18], acc[19]);
                float2 q_136 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                float2 r_137 = fma_f32x2_rn_noftz(q_136, scale1_107, a_135);
                acc[18] = r_137.x;
                acc[19] = r_137.y;
                float2 a_138 = make_float2(acc[20], acc[21]);
                float2 q_139 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                float2 r_140 = fma_f32x2_rn_noftz(q_139, scale1_107, a_138);
                acc[20] = r_140.x;
                acc[21] = r_140.y;
                float2 a_141 = make_float2(acc[22], acc[23]);
                float2 q_142 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                float2 r_143 = fma_f32x2_rn_noftz(q_142, scale1_107, a_141);
                acc[22] = r_143.x;
                acc[23] = r_143.y;
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
                float2 a_144 = make_float2(acc[24], acc[25]);
                float2 q_145 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                float2 r_146 = fma_f32x2_rn_noftz(q_145, scale1_107, a_144);
                acc[24] = r_146.x;
                acc[25] = r_146.y;
                float2 a_147 = make_float2(acc[26], acc[27]);
                float2 q_148 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                float2 r_149 = fma_f32x2_rn_noftz(q_148, scale1_107, a_147);
                acc[26] = r_149.x;
                acc[27] = r_149.y;
                float2 a_150 = make_float2(acc[28], acc[29]);
                float2 q_151 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                float2 r_152 = fma_f32x2_rn_noftz(q_151, scale1_107, a_150);
                acc[28] = r_152.x;
                acc[29] = r_152.y;
                float2 a_153 = make_float2(acc[30], acc[31]);
                float2 q_154 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                float2 r_155 = fma_f32x2_rn_noftz(q_154, scale1_107, a_153);
                acc[30] = r_155.x;
                acc[31] = r_155.y;
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
                float2 scale0_156 = make_float2(_mul_6, _mul_6);
                float2 scale1_157 = make_float2(_mul_7, _mul_7);
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
                float2 a_158 = make_float2(acc[0], acc[1]);
                float2 q_159 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                float2 r_160 = fma_f32x2_rn_noftz(q_159, scale0_156, a_158);
                acc[0] = r_160.x;
                acc[1] = r_160.y;
                float2 a_161 = make_float2(acc[2], acc[3]);
                float2 q_162 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                float2 r_163 = fma_f32x2_rn_noftz(q_162, scale0_156, a_161);
                acc[2] = r_163.x;
                acc[3] = r_163.y;
                float2 a_164 = make_float2(acc[4], acc[5]);
                float2 q_165 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                float2 r_166 = fma_f32x2_rn_noftz(q_165, scale0_156, a_164);
                acc[4] = r_166.x;
                acc[5] = r_166.y;
                float2 a_167 = make_float2(acc[6], acc[7]);
                float2 q_168 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                float2 r_169 = fma_f32x2_rn_noftz(q_168, scale0_156, a_167);
                acc[6] = r_169.x;
                acc[7] = r_169.y;
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
                float2 a_170 = make_float2(acc[8], acc[9]);
                float2 q_171 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                float2 r_172 = fma_f32x2_rn_noftz(q_171, scale0_156, a_170);
                acc[8] = r_172.x;
                acc[9] = r_172.y;
                float2 a_173 = make_float2(acc[10], acc[11]);
                float2 q_174 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                float2 r_175 = fma_f32x2_rn_noftz(q_174, scale0_156, a_173);
                acc[10] = r_175.x;
                acc[11] = r_175.y;
                float2 a_176 = make_float2(acc[12], acc[13]);
                float2 q_177 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                float2 r_178 = fma_f32x2_rn_noftz(q_177, scale0_156, a_176);
                acc[12] = r_178.x;
                acc[13] = r_178.y;
                float2 a_179 = make_float2(acc[14], acc[15]);
                float2 q_180 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                float2 r_181 = fma_f32x2_rn_noftz(q_180, scale0_156, a_179);
                acc[14] = r_181.x;
                acc[15] = r_181.y;
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
                float2 a_182 = make_float2(acc[16], acc[17]);
                float2 q_183 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                float2 r_184 = fma_f32x2_rn_noftz(q_183, scale1_157, a_182);
                acc[16] = r_184.x;
                acc[17] = r_184.y;
                float2 a_185 = make_float2(acc[18], acc[19]);
                float2 q_186 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                float2 r_187 = fma_f32x2_rn_noftz(q_186, scale1_157, a_185);
                acc[18] = r_187.x;
                acc[19] = r_187.y;
                float2 a_188 = make_float2(acc[20], acc[21]);
                float2 q_189 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                float2 r_190 = fma_f32x2_rn_noftz(q_189, scale1_157, a_188);
                acc[20] = r_190.x;
                acc[21] = r_190.y;
                float2 a_191 = make_float2(acc[22], acc[23]);
                float2 q_192 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                float2 r_193 = fma_f32x2_rn_noftz(q_192, scale1_157, a_191);
                acc[22] = r_193.x;
                acc[23] = r_193.y;
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
                float2 a_194 = make_float2(acc[24], acc[25]);
                float2 q_195 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                float2 r_196 = fma_f32x2_rn_noftz(q_195, scale1_157, a_194);
                acc[24] = r_196.x;
                acc[25] = r_196.y;
                float2 a_197 = make_float2(acc[26], acc[27]);
                float2 q_198 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                float2 r_199 = fma_f32x2_rn_noftz(q_198, scale1_157, a_197);
                acc[26] = r_199.x;
                acc[27] = r_199.y;
                float2 a_200 = make_float2(acc[28], acc[29]);
                float2 q_201 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                float2 r_202 = fma_f32x2_rn_noftz(q_201, scale1_157, a_200);
                acc[28] = r_202.x;
                acc[29] = r_202.y;
                float2 a_203 = make_float2(acc[30], acc[31]);
                float2 q_204 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                float2 r_205 = fma_f32x2_rn_noftz(q_204, scale1_157, a_203);
                acc[30] = r_205.x;
                acc[31] = r_205.y;
                float acc_206[32] = {0};
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
                float2 scale0_207 = make_float2(_mul_8, _mul_8);
                float2 scale1_208 = make_float2(_mul_9, _mul_9);
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
                float2 a_209 = make_float2(acc_206[0], acc_206[1]);
                float2 q_210 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                float2 r_211 = fma_f32x2_rn_noftz(q_210, scale0_207, a_209);
                acc_206[0] = r_211.x;
                acc_206[1] = r_211.y;
                float2 a_212 = make_float2(acc_206[2], acc_206[3]);
                float2 q_213 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                float2 r_214 = fma_f32x2_rn_noftz(q_213, scale0_207, a_212);
                acc_206[2] = r_214.x;
                acc_206[3] = r_214.y;
                float2 a_215 = make_float2(acc_206[4], acc_206[5]);
                float2 q_216 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                float2 r_217 = fma_f32x2_rn_noftz(q_216, scale0_207, a_215);
                acc_206[4] = r_217.x;
                acc_206[5] = r_217.y;
                float2 a_218 = make_float2(acc_206[6], acc_206[7]);
                float2 q_219 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                float2 r_220 = fma_f32x2_rn_noftz(q_219, scale0_207, a_218);
                acc_206[6] = r_220.x;
                acc_206[7] = r_220.y;
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
                float2 a_221 = make_float2(acc_206[8], acc_206[9]);
                float2 q_222 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_207, a_221);
                acc_206[8] = r_223.x;
                acc_206[9] = r_223.y;
                float2 a_224 = make_float2(acc_206[10], acc_206[11]);
                float2 q_225 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                float2 r_226 = fma_f32x2_rn_noftz(q_225, scale0_207, a_224);
                acc_206[10] = r_226.x;
                acc_206[11] = r_226.y;
                float2 a_227 = make_float2(acc_206[12], acc_206[13]);
                float2 q_228 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                float2 r_229 = fma_f32x2_rn_noftz(q_228, scale0_207, a_227);
                acc_206[12] = r_229.x;
                acc_206[13] = r_229.y;
                float2 a_230 = make_float2(acc_206[14], acc_206[15]);
                float2 q_231 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                float2 r_232 = fma_f32x2_rn_noftz(q_231, scale0_207, a_230);
                acc_206[14] = r_232.x;
                acc_206[15] = r_232.y;
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
                float2 a_233 = make_float2(acc_206[16], acc_206[17]);
                float2 q_234 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                float2 r_235 = fma_f32x2_rn_noftz(q_234, scale1_208, a_233);
                acc_206[16] = r_235.x;
                acc_206[17] = r_235.y;
                float2 a_236 = make_float2(acc_206[18], acc_206[19]);
                float2 q_237 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                float2 r_238 = fma_f32x2_rn_noftz(q_237, scale1_208, a_236);
                acc_206[18] = r_238.x;
                acc_206[19] = r_238.y;
                float2 a_239 = make_float2(acc_206[20], acc_206[21]);
                float2 q_240 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                float2 r_241 = fma_f32x2_rn_noftz(q_240, scale1_208, a_239);
                acc_206[20] = r_241.x;
                acc_206[21] = r_241.y;
                float2 a_242 = make_float2(acc_206[22], acc_206[23]);
                float2 q_243 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                float2 r_244 = fma_f32x2_rn_noftz(q_243, scale1_208, a_242);
                acc_206[22] = r_244.x;
                acc_206[23] = r_244.y;
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
                float2 a_245 = make_float2(acc_206[24], acc_206[25]);
                float2 q_246 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_208, a_245);
                acc_206[24] = r_247.x;
                acc_206[25] = r_247.y;
                float2 a_248 = make_float2(acc_206[26], acc_206[27]);
                float2 q_249 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                float2 r_250 = fma_f32x2_rn_noftz(q_249, scale1_208, a_248);
                acc_206[26] = r_250.x;
                acc_206[27] = r_250.y;
                float2 a_251 = make_float2(acc_206[28], acc_206[29]);
                float2 q_252 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                float2 r_253 = fma_f32x2_rn_noftz(q_252, scale1_208, a_251);
                acc_206[28] = r_253.x;
                acc_206[29] = r_253.y;
                float2 a_254 = make_float2(acc_206[30], acc_206[31]);
                float2 q_255 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                float2 r_256 = fma_f32x2_rn_noftz(q_255, scale1_208, a_254);
                acc_206[30] = r_256.x;
                acc_206[31] = r_256.y;
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
                float2 scale0_257 = make_float2(_mul_10, _mul_10);
                float2 scale1_258 = make_float2(_mul_11, _mul_11);
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
                float2 a_259 = make_float2(acc_206[0], acc_206[1]);
                float2 q_260 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                float2 r_261 = fma_f32x2_rn_noftz(q_260, scale0_257, a_259);
                acc_206[0] = r_261.x;
                acc_206[1] = r_261.y;
                float2 a_262 = make_float2(acc_206[2], acc_206[3]);
                float2 q_263 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                float2 r_264 = fma_f32x2_rn_noftz(q_263, scale0_257, a_262);
                acc_206[2] = r_264.x;
                acc_206[3] = r_264.y;
                float2 a_265 = make_float2(acc_206[4], acc_206[5]);
                float2 q_266 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                float2 r_267 = fma_f32x2_rn_noftz(q_266, scale0_257, a_265);
                acc_206[4] = r_267.x;
                acc_206[5] = r_267.y;
                float2 a_268 = make_float2(acc_206[6], acc_206[7]);
                float2 q_269 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                float2 r_270 = fma_f32x2_rn_noftz(q_269, scale0_257, a_268);
                acc_206[6] = r_270.x;
                acc_206[7] = r_270.y;
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
                float2 a_271 = make_float2(acc_206[8], acc_206[9]);
                float2 q_272 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                float2 r_273 = fma_f32x2_rn_noftz(q_272, scale0_257, a_271);
                acc_206[8] = r_273.x;
                acc_206[9] = r_273.y;
                float2 a_274 = make_float2(acc_206[10], acc_206[11]);
                float2 q_275 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                float2 r_276 = fma_f32x2_rn_noftz(q_275, scale0_257, a_274);
                acc_206[10] = r_276.x;
                acc_206[11] = r_276.y;
                float2 a_277 = make_float2(acc_206[12], acc_206[13]);
                float2 q_278 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                float2 r_279 = fma_f32x2_rn_noftz(q_278, scale0_257, a_277);
                acc_206[12] = r_279.x;
                acc_206[13] = r_279.y;
                float2 a_280 = make_float2(acc_206[14], acc_206[15]);
                float2 q_281 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                float2 r_282 = fma_f32x2_rn_noftz(q_281, scale0_257, a_280);
                acc_206[14] = r_282.x;
                acc_206[15] = r_282.y;
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
                float2 a_283 = make_float2(acc_206[16], acc_206[17]);
                float2 q_284 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                float2 r_285 = fma_f32x2_rn_noftz(q_284, scale1_258, a_283);
                acc_206[16] = r_285.x;
                acc_206[17] = r_285.y;
                float2 a_286 = make_float2(acc_206[18], acc_206[19]);
                float2 q_287 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                float2 r_288 = fma_f32x2_rn_noftz(q_287, scale1_258, a_286);
                acc_206[18] = r_288.x;
                acc_206[19] = r_288.y;
                float2 a_289 = make_float2(acc_206[20], acc_206[21]);
                float2 q_290 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                float2 r_291 = fma_f32x2_rn_noftz(q_290, scale1_258, a_289);
                acc_206[20] = r_291.x;
                acc_206[21] = r_291.y;
                float2 a_292 = make_float2(acc_206[22], acc_206[23]);
                float2 q_293 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                float2 r_294 = fma_f32x2_rn_noftz(q_293, scale1_258, a_292);
                acc_206[22] = r_294.x;
                acc_206[23] = r_294.y;
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
                float2 a_295 = make_float2(acc_206[24], acc_206[25]);
                float2 q_296 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                float2 r_297 = fma_f32x2_rn_noftz(q_296, scale1_258, a_295);
                acc_206[24] = r_297.x;
                acc_206[25] = r_297.y;
                float2 a_298 = make_float2(acc_206[26], acc_206[27]);
                float2 q_299 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                float2 r_300 = fma_f32x2_rn_noftz(q_299, scale1_258, a_298);
                acc_206[26] = r_300.x;
                acc_206[27] = r_300.y;
                float2 a_301 = make_float2(acc_206[28], acc_206[29]);
                float2 q_302 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                float2 r_303 = fma_f32x2_rn_noftz(q_302, scale1_258, a_301);
                acc_206[28] = r_303.x;
                acc_206[29] = r_303.y;
                float2 a_304 = make_float2(acc_206[30], acc_206[31]);
                float2 q_305 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                float2 r_306 = fma_f32x2_rn_noftz(q_305, scale1_258, a_304);
                acc_206[30] = r_306.x;
                acc_206[31] = r_306.y;
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
                float2 scale0_307 = make_float2(_mul_12, _mul_12);
                float2 scale1_308 = make_float2(_mul_13, _mul_13);
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
                float2 a_309 = make_float2(acc_206[0], acc_206[1]);
                float2 q_310 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                float2 r_311 = fma_f32x2_rn_noftz(q_310, scale0_307, a_309);
                acc_206[0] = r_311.x;
                acc_206[1] = r_311.y;
                float2 a_312 = make_float2(acc_206[2], acc_206[3]);
                float2 q_313 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                float2 r_314 = fma_f32x2_rn_noftz(q_313, scale0_307, a_312);
                acc_206[2] = r_314.x;
                acc_206[3] = r_314.y;
                float2 a_315 = make_float2(acc_206[4], acc_206[5]);
                float2 q_316 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                float2 r_317 = fma_f32x2_rn_noftz(q_316, scale0_307, a_315);
                acc_206[4] = r_317.x;
                acc_206[5] = r_317.y;
                float2 a_318 = make_float2(acc_206[6], acc_206[7]);
                float2 q_319 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                float2 r_320 = fma_f32x2_rn_noftz(q_319, scale0_307, a_318);
                acc_206[6] = r_320.x;
                acc_206[7] = r_320.y;
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
                float2 a_321 = make_float2(acc_206[8], acc_206[9]);
                float2 q_322 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                float2 r_323 = fma_f32x2_rn_noftz(q_322, scale0_307, a_321);
                acc_206[8] = r_323.x;
                acc_206[9] = r_323.y;
                float2 a_324 = make_float2(acc_206[10], acc_206[11]);
                float2 q_325 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                float2 r_326 = fma_f32x2_rn_noftz(q_325, scale0_307, a_324);
                acc_206[10] = r_326.x;
                acc_206[11] = r_326.y;
                float2 a_327 = make_float2(acc_206[12], acc_206[13]);
                float2 q_328 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                float2 r_329 = fma_f32x2_rn_noftz(q_328, scale0_307, a_327);
                acc_206[12] = r_329.x;
                acc_206[13] = r_329.y;
                float2 a_330 = make_float2(acc_206[14], acc_206[15]);
                float2 q_331 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                float2 r_332 = fma_f32x2_rn_noftz(q_331, scale0_307, a_330);
                acc_206[14] = r_332.x;
                acc_206[15] = r_332.y;
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
                float2 a_333 = make_float2(acc_206[16], acc_206[17]);
                float2 q_334 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                float2 r_335 = fma_f32x2_rn_noftz(q_334, scale1_308, a_333);
                acc_206[16] = r_335.x;
                acc_206[17] = r_335.y;
                float2 a_336 = make_float2(acc_206[18], acc_206[19]);
                float2 q_337 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                float2 r_338 = fma_f32x2_rn_noftz(q_337, scale1_308, a_336);
                acc_206[18] = r_338.x;
                acc_206[19] = r_338.y;
                float2 a_339 = make_float2(acc_206[20], acc_206[21]);
                float2 q_340 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                float2 r_341 = fma_f32x2_rn_noftz(q_340, scale1_308, a_339);
                acc_206[20] = r_341.x;
                acc_206[21] = r_341.y;
                float2 a_342 = make_float2(acc_206[22], acc_206[23]);
                float2 q_343 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                float2 r_344 = fma_f32x2_rn_noftz(q_343, scale1_308, a_342);
                acc_206[22] = r_344.x;
                acc_206[23] = r_344.y;
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
                float2 a_345 = make_float2(acc_206[24], acc_206[25]);
                float2 q_346 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                float2 r_347 = fma_f32x2_rn_noftz(q_346, scale1_308, a_345);
                acc_206[24] = r_347.x;
                acc_206[25] = r_347.y;
                float2 a_348 = make_float2(acc_206[26], acc_206[27]);
                float2 q_349 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                float2 r_350 = fma_f32x2_rn_noftz(q_349, scale1_308, a_348);
                acc_206[26] = r_350.x;
                acc_206[27] = r_350.y;
                float2 a_351 = make_float2(acc_206[28], acc_206[29]);
                float2 q_352 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                float2 r_353 = fma_f32x2_rn_noftz(q_352, scale1_308, a_351);
                acc_206[28] = r_353.x;
                acc_206[29] = r_353.y;
                float2 a_354 = make_float2(acc_206[30], acc_206[31]);
                float2 q_355 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                float2 r_356 = fma_f32x2_rn_noftz(q_355, scale1_308, a_354);
                acc_206[30] = r_356.x;
                acc_206[31] = r_356.y;
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
                float2 scale0_357 = make_float2(_mul_14, _mul_14);
                float2 scale1_358 = make_float2(_mul_15, _mul_15);
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
                float2 a_359 = make_float2(acc_206[0], acc_206[1]);
                float2 q_360 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                float2 r_361 = fma_f32x2_rn_noftz(q_360, scale0_357, a_359);
                acc_206[0] = r_361.x;
                acc_206[1] = r_361.y;
                float2 a_362 = make_float2(acc_206[2], acc_206[3]);
                float2 q_363 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                float2 r_364 = fma_f32x2_rn_noftz(q_363, scale0_357, a_362);
                acc_206[2] = r_364.x;
                acc_206[3] = r_364.y;
                float2 a_365 = make_float2(acc_206[4], acc_206[5]);
                float2 q_366 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                float2 r_367 = fma_f32x2_rn_noftz(q_366, scale0_357, a_365);
                acc_206[4] = r_367.x;
                acc_206[5] = r_367.y;
                float2 a_368 = make_float2(acc_206[6], acc_206[7]);
                float2 q_369 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                float2 r_370 = fma_f32x2_rn_noftz(q_369, scale0_357, a_368);
                acc_206[6] = r_370.x;
                acc_206[7] = r_370.y;
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
                float2 a_371 = make_float2(acc_206[8], acc_206[9]);
                float2 q_372 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                float2 r_373 = fma_f32x2_rn_noftz(q_372, scale0_357, a_371);
                acc_206[8] = r_373.x;
                acc_206[9] = r_373.y;
                float2 a_374 = make_float2(acc_206[10], acc_206[11]);
                float2 q_375 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                float2 r_376 = fma_f32x2_rn_noftz(q_375, scale0_357, a_374);
                acc_206[10] = r_376.x;
                acc_206[11] = r_376.y;
                float2 a_377 = make_float2(acc_206[12], acc_206[13]);
                float2 q_378 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                float2 r_379 = fma_f32x2_rn_noftz(q_378, scale0_357, a_377);
                acc_206[12] = r_379.x;
                acc_206[13] = r_379.y;
                float2 a_380 = make_float2(acc_206[14], acc_206[15]);
                float2 q_381 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                float2 r_382 = fma_f32x2_rn_noftz(q_381, scale0_357, a_380);
                acc_206[14] = r_382.x;
                acc_206[15] = r_382.y;
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
                float2 a_383 = make_float2(acc_206[16], acc_206[17]);
                float2 q_384 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                float2 r_385 = fma_f32x2_rn_noftz(q_384, scale1_358, a_383);
                acc_206[16] = r_385.x;
                acc_206[17] = r_385.y;
                float2 a_386 = make_float2(acc_206[18], acc_206[19]);
                float2 q_387 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                float2 r_388 = fma_f32x2_rn_noftz(q_387, scale1_358, a_386);
                acc_206[18] = r_388.x;
                acc_206[19] = r_388.y;
                float2 a_389 = make_float2(acc_206[20], acc_206[21]);
                float2 q_390 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                float2 r_391 = fma_f32x2_rn_noftz(q_390, scale1_358, a_389);
                acc_206[20] = r_391.x;
                acc_206[21] = r_391.y;
                float2 a_392 = make_float2(acc_206[22], acc_206[23]);
                float2 q_393 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                float2 r_394 = fma_f32x2_rn_noftz(q_393, scale1_358, a_392);
                acc_206[22] = r_394.x;
                acc_206[23] = r_394.y;
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
                float2 a_395 = make_float2(acc_206[24], acc_206[25]);
                float2 q_396 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                float2 r_397 = fma_f32x2_rn_noftz(q_396, scale1_358, a_395);
                acc_206[24] = r_397.x;
                acc_206[25] = r_397.y;
                float2 a_398 = make_float2(acc_206[26], acc_206[27]);
                float2 q_399 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                float2 r_400 = fma_f32x2_rn_noftz(q_399, scale1_358, a_398);
                acc_206[26] = r_400.x;
                acc_206[27] = r_400.y;
                float2 a_401 = make_float2(acc_206[28], acc_206[29]);
                float2 q_402 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                float2 r_403 = fma_f32x2_rn_noftz(q_402, scale1_358, a_401);
                acc_206[28] = r_403.x;
                acc_206[29] = r_403.y;
                float2 a_404 = make_float2(acc_206[30], acc_206[31]);
                float2 q_405 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                float2 r_406 = fma_f32x2_rn_noftz(q_405, scale1_358, a_404);
                acc_206[30] = r_406.x;
                acc_206[31] = r_406.y;
                float acc_407[32] = {0};
                uint32_t _e4m3x2_to_f16x2_14;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_14) : "h"((uint16_t)(_fp4_wire_scales_0[8])));
                float _cvt_f32_f16_28;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(_e4m3x2_to_f16x2_14)));
                float _cvt_f32_f16_29;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(_e4m3x2_to_f16x2_14 >> 16)));
                float _mul_16;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_28), "f"(rs_s));
                float _mul_17;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_29), "f"(rs_s));
                float2 scale0_408 = make_float2(_mul_16, _mul_16);
                float2 scale1_409 = make_float2(_mul_17, _mul_17);
                float _unpack_fp4_32[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[32 + 0]));
                    (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(_fp4_wire_values_0[32 + 0]));
                    #endif
                }
                float2 a_410 = make_float2(acc_407[0], acc_407[1]);
                float2 q_411 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                float2 r_412 = fma_f32x2_rn_noftz(q_411, scale0_408, a_410);
                acc_407[0] = r_412.x;
                acc_407[1] = r_412.y;
                float2 a_413 = make_float2(acc_407[2], acc_407[3]);
                float2 q_414 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                float2 r_415 = fma_f32x2_rn_noftz(q_414, scale0_408, a_413);
                acc_407[2] = r_415.x;
                acc_407[3] = r_415.y;
                float2 a_416 = make_float2(acc_407[4], acc_407[5]);
                float2 q_417 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                float2 r_418 = fma_f32x2_rn_noftz(q_417, scale0_408, a_416);
                acc_407[4] = r_418.x;
                acc_407[5] = r_418.y;
                float2 a_419 = make_float2(acc_407[6], acc_407[7]);
                float2 q_420 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                float2 r_421 = fma_f32x2_rn_noftz(q_420, scale0_408, a_419);
                acc_407[6] = r_421.x;
                acc_407[7] = r_421.y;
                float _unpack_fp4_33[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[33 + 0]));
                    (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(_fp4_wire_values_0[33 + 0]));
                    #endif
                }
                float2 a_422 = make_float2(acc_407[8], acc_407[9]);
                float2 q_423 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                float2 r_424 = fma_f32x2_rn_noftz(q_423, scale0_408, a_422);
                acc_407[8] = r_424.x;
                acc_407[9] = r_424.y;
                float2 a_425 = make_float2(acc_407[10], acc_407[11]);
                float2 q_426 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                float2 r_427 = fma_f32x2_rn_noftz(q_426, scale0_408, a_425);
                acc_407[10] = r_427.x;
                acc_407[11] = r_427.y;
                float2 a_428 = make_float2(acc_407[12], acc_407[13]);
                float2 q_429 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                float2 r_430 = fma_f32x2_rn_noftz(q_429, scale0_408, a_428);
                acc_407[12] = r_430.x;
                acc_407[13] = r_430.y;
                float2 a_431 = make_float2(acc_407[14], acc_407[15]);
                float2 q_432 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                float2 r_433 = fma_f32x2_rn_noftz(q_432, scale0_408, a_431);
                acc_407[14] = r_433.x;
                acc_407[15] = r_433.y;
                float _unpack_fp4_34[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[34 + 0]));
                    (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(_fp4_wire_values_0[34 + 0]));
                    #endif
                }
                float2 a_434 = make_float2(acc_407[16], acc_407[17]);
                float2 q_435 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                float2 r_436 = fma_f32x2_rn_noftz(q_435, scale1_409, a_434);
                acc_407[16] = r_436.x;
                acc_407[17] = r_436.y;
                float2 a_437 = make_float2(acc_407[18], acc_407[19]);
                float2 q_438 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                float2 r_439 = fma_f32x2_rn_noftz(q_438, scale1_409, a_437);
                acc_407[18] = r_439.x;
                acc_407[19] = r_439.y;
                float2 a_440 = make_float2(acc_407[20], acc_407[21]);
                float2 q_441 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                float2 r_442 = fma_f32x2_rn_noftz(q_441, scale1_409, a_440);
                acc_407[20] = r_442.x;
                acc_407[21] = r_442.y;
                float2 a_443 = make_float2(acc_407[22], acc_407[23]);
                float2 q_444 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                float2 r_445 = fma_f32x2_rn_noftz(q_444, scale1_409, a_443);
                acc_407[22] = r_445.x;
                acc_407[23] = r_445.y;
                float _unpack_fp4_35[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[35 + 0]));
                    (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(_fp4_wire_values_0[35 + 0]));
                    #endif
                }
                float2 a_446 = make_float2(acc_407[24], acc_407[25]);
                float2 q_447 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                float2 r_448 = fma_f32x2_rn_noftz(q_447, scale1_409, a_446);
                acc_407[24] = r_448.x;
                acc_407[25] = r_448.y;
                float2 a_449 = make_float2(acc_407[26], acc_407[27]);
                float2 q_450 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                float2 r_451 = fma_f32x2_rn_noftz(q_450, scale1_409, a_449);
                acc_407[26] = r_451.x;
                acc_407[27] = r_451.y;
                float2 a_452 = make_float2(acc_407[28], acc_407[29]);
                float2 q_453 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                float2 r_454 = fma_f32x2_rn_noftz(q_453, scale1_409, a_452);
                acc_407[28] = r_454.x;
                acc_407[29] = r_454.y;
                float2 a_455 = make_float2(acc_407[30], acc_407[31]);
                float2 q_456 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                float2 r_457 = fma_f32x2_rn_noftz(q_456, scale1_409, a_455);
                acc_407[30] = r_457.x;
                acc_407[31] = r_457.y;
                uint32_t _e4m3x2_to_f16x2_15;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_15) : "h"((uint16_t)(_fp4_wire_scales_0[9])));
                float _cvt_f32_f16_30;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(_e4m3x2_to_f16x2_15)));
                float _cvt_f32_f16_31;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(_e4m3x2_to_f16x2_15 >> 16)));
                float _mul_18;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_30), "f"(rs_s_0));
                float _mul_19;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_31), "f"(rs_s_0));
                float2 scale0_458 = make_float2(_mul_18, _mul_18);
                float2 scale1_459 = make_float2(_mul_19, _mul_19);
                float _unpack_fp4_36[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[36 + 0]));
                    (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(_fp4_wire_values_0[36 + 0]));
                    #endif
                }
                float2 a_460 = make_float2(acc_407[0], acc_407[1]);
                float2 q_461 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                float2 r_462 = fma_f32x2_rn_noftz(q_461, scale0_458, a_460);
                acc_407[0] = r_462.x;
                acc_407[1] = r_462.y;
                float2 a_463 = make_float2(acc_407[2], acc_407[3]);
                float2 q_464 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                float2 r_465 = fma_f32x2_rn_noftz(q_464, scale0_458, a_463);
                acc_407[2] = r_465.x;
                acc_407[3] = r_465.y;
                float2 a_466 = make_float2(acc_407[4], acc_407[5]);
                float2 q_467 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                float2 r_468 = fma_f32x2_rn_noftz(q_467, scale0_458, a_466);
                acc_407[4] = r_468.x;
                acc_407[5] = r_468.y;
                float2 a_469 = make_float2(acc_407[6], acc_407[7]);
                float2 q_470 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                float2 r_471 = fma_f32x2_rn_noftz(q_470, scale0_458, a_469);
                acc_407[6] = r_471.x;
                acc_407[7] = r_471.y;
                float _unpack_fp4_37[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[37 + 0]));
                    (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(_fp4_wire_values_0[37 + 0]));
                    #endif
                }
                float2 a_472 = make_float2(acc_407[8], acc_407[9]);
                float2 q_473 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                float2 r_474 = fma_f32x2_rn_noftz(q_473, scale0_458, a_472);
                acc_407[8] = r_474.x;
                acc_407[9] = r_474.y;
                float2 a_475 = make_float2(acc_407[10], acc_407[11]);
                float2 q_476 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                float2 r_477 = fma_f32x2_rn_noftz(q_476, scale0_458, a_475);
                acc_407[10] = r_477.x;
                acc_407[11] = r_477.y;
                float2 a_478 = make_float2(acc_407[12], acc_407[13]);
                float2 q_479 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                float2 r_480 = fma_f32x2_rn_noftz(q_479, scale0_458, a_478);
                acc_407[12] = r_480.x;
                acc_407[13] = r_480.y;
                float2 a_481 = make_float2(acc_407[14], acc_407[15]);
                float2 q_482 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                float2 r_483 = fma_f32x2_rn_noftz(q_482, scale0_458, a_481);
                acc_407[14] = r_483.x;
                acc_407[15] = r_483.y;
                float _unpack_fp4_38[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[38 + 0]));
                    (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(_fp4_wire_values_0[38 + 0]));
                    #endif
                }
                float2 a_484 = make_float2(acc_407[16], acc_407[17]);
                float2 q_485 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                float2 r_486 = fma_f32x2_rn_noftz(q_485, scale1_459, a_484);
                acc_407[16] = r_486.x;
                acc_407[17] = r_486.y;
                float2 a_487 = make_float2(acc_407[18], acc_407[19]);
                float2 q_488 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                float2 r_489 = fma_f32x2_rn_noftz(q_488, scale1_459, a_487);
                acc_407[18] = r_489.x;
                acc_407[19] = r_489.y;
                float2 a_490 = make_float2(acc_407[20], acc_407[21]);
                float2 q_491 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                float2 r_492 = fma_f32x2_rn_noftz(q_491, scale1_459, a_490);
                acc_407[20] = r_492.x;
                acc_407[21] = r_492.y;
                float2 a_493 = make_float2(acc_407[22], acc_407[23]);
                float2 q_494 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                float2 r_495 = fma_f32x2_rn_noftz(q_494, scale1_459, a_493);
                acc_407[22] = r_495.x;
                acc_407[23] = r_495.y;
                float _unpack_fp4_39[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[39 + 0]));
                    (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(_fp4_wire_values_0[39 + 0]));
                    #endif
                }
                float2 a_496 = make_float2(acc_407[24], acc_407[25]);
                float2 q_497 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                float2 r_498 = fma_f32x2_rn_noftz(q_497, scale1_459, a_496);
                acc_407[24] = r_498.x;
                acc_407[25] = r_498.y;
                float2 a_499 = make_float2(acc_407[26], acc_407[27]);
                float2 q_500 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                float2 r_501 = fma_f32x2_rn_noftz(q_500, scale1_459, a_499);
                acc_407[26] = r_501.x;
                acc_407[27] = r_501.y;
                float2 a_502 = make_float2(acc_407[28], acc_407[29]);
                float2 q_503 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                float2 r_504 = fma_f32x2_rn_noftz(q_503, scale1_459, a_502);
                acc_407[28] = r_504.x;
                acc_407[29] = r_504.y;
                float2 a_505 = make_float2(acc_407[30], acc_407[31]);
                float2 q_506 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                float2 r_507 = fma_f32x2_rn_noftz(q_506, scale1_459, a_505);
                acc_407[30] = r_507.x;
                acc_407[31] = r_507.y;
                uint32_t _e4m3x2_to_f16x2_16;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_16) : "h"((uint16_t)(_fp4_wire_scales_0[10])));
                float _cvt_f32_f16_32;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_32) : "h"((uint16_t)(_e4m3x2_to_f16x2_16)));
                float _cvt_f32_f16_33;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_33) : "h"((uint16_t)(_e4m3x2_to_f16x2_16 >> 16)));
                float _mul_20;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_32), "f"(rs_s_1));
                float _mul_21;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_33), "f"(rs_s_1));
                float2 scale0_508 = make_float2(_mul_20, _mul_20);
                float2 scale1_509 = make_float2(_mul_21, _mul_21);
                float _unpack_fp4_40[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[40 + 0]));
                    (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(_fp4_wire_values_0[40 + 0]));
                    #endif
                }
                float2 a_510 = make_float2(acc_407[0], acc_407[1]);
                float2 q_511 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                float2 r_512 = fma_f32x2_rn_noftz(q_511, scale0_508, a_510);
                acc_407[0] = r_512.x;
                acc_407[1] = r_512.y;
                float2 a_513 = make_float2(acc_407[2], acc_407[3]);
                float2 q_514 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                float2 r_515 = fma_f32x2_rn_noftz(q_514, scale0_508, a_513);
                acc_407[2] = r_515.x;
                acc_407[3] = r_515.y;
                float2 a_516 = make_float2(acc_407[4], acc_407[5]);
                float2 q_517 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                float2 r_518 = fma_f32x2_rn_noftz(q_517, scale0_508, a_516);
                acc_407[4] = r_518.x;
                acc_407[5] = r_518.y;
                float2 a_519 = make_float2(acc_407[6], acc_407[7]);
                float2 q_520 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                float2 r_521 = fma_f32x2_rn_noftz(q_520, scale0_508, a_519);
                acc_407[6] = r_521.x;
                acc_407[7] = r_521.y;
                float _unpack_fp4_41[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[41 + 0]));
                    (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(_fp4_wire_values_0[41 + 0]));
                    #endif
                }
                float2 a_522 = make_float2(acc_407[8], acc_407[9]);
                float2 q_523 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                float2 r_524 = fma_f32x2_rn_noftz(q_523, scale0_508, a_522);
                acc_407[8] = r_524.x;
                acc_407[9] = r_524.y;
                float2 a_525 = make_float2(acc_407[10], acc_407[11]);
                float2 q_526 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                float2 r_527 = fma_f32x2_rn_noftz(q_526, scale0_508, a_525);
                acc_407[10] = r_527.x;
                acc_407[11] = r_527.y;
                float2 a_528 = make_float2(acc_407[12], acc_407[13]);
                float2 q_529 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                float2 r_530 = fma_f32x2_rn_noftz(q_529, scale0_508, a_528);
                acc_407[12] = r_530.x;
                acc_407[13] = r_530.y;
                float2 a_531 = make_float2(acc_407[14], acc_407[15]);
                float2 q_532 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                float2 r_533 = fma_f32x2_rn_noftz(q_532, scale0_508, a_531);
                acc_407[14] = r_533.x;
                acc_407[15] = r_533.y;
                float _unpack_fp4_42[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[42 + 0]));
                    (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(_fp4_wire_values_0[42 + 0]));
                    #endif
                }
                float2 a_534 = make_float2(acc_407[16], acc_407[17]);
                float2 q_535 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                float2 r_536 = fma_f32x2_rn_noftz(q_535, scale1_509, a_534);
                acc_407[16] = r_536.x;
                acc_407[17] = r_536.y;
                float2 a_537 = make_float2(acc_407[18], acc_407[19]);
                float2 q_538 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                float2 r_539 = fma_f32x2_rn_noftz(q_538, scale1_509, a_537);
                acc_407[18] = r_539.x;
                acc_407[19] = r_539.y;
                float2 a_540 = make_float2(acc_407[20], acc_407[21]);
                float2 q_541 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                float2 r_542 = fma_f32x2_rn_noftz(q_541, scale1_509, a_540);
                acc_407[20] = r_542.x;
                acc_407[21] = r_542.y;
                float2 a_543 = make_float2(acc_407[22], acc_407[23]);
                float2 q_544 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                float2 r_545 = fma_f32x2_rn_noftz(q_544, scale1_509, a_543);
                acc_407[22] = r_545.x;
                acc_407[23] = r_545.y;
                float _unpack_fp4_43[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[43 + 0]));
                    (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(_fp4_wire_values_0[43 + 0]));
                    #endif
                }
                float2 a_546 = make_float2(acc_407[24], acc_407[25]);
                float2 q_547 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                float2 r_548 = fma_f32x2_rn_noftz(q_547, scale1_509, a_546);
                acc_407[24] = r_548.x;
                acc_407[25] = r_548.y;
                float2 a_549 = make_float2(acc_407[26], acc_407[27]);
                float2 q_550 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                float2 r_551 = fma_f32x2_rn_noftz(q_550, scale1_509, a_549);
                acc_407[26] = r_551.x;
                acc_407[27] = r_551.y;
                float2 a_552 = make_float2(acc_407[28], acc_407[29]);
                float2 q_553 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                float2 r_554 = fma_f32x2_rn_noftz(q_553, scale1_509, a_552);
                acc_407[28] = r_554.x;
                acc_407[29] = r_554.y;
                float2 a_555 = make_float2(acc_407[30], acc_407[31]);
                float2 q_556 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                float2 r_557 = fma_f32x2_rn_noftz(q_556, scale1_509, a_555);
                acc_407[30] = r_557.x;
                acc_407[31] = r_557.y;
                uint32_t _e4m3x2_to_f16x2_17;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_17) : "h"((uint16_t)(_fp4_wire_scales_0[11])));
                float _cvt_f32_f16_34;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_34) : "h"((uint16_t)(_e4m3x2_to_f16x2_17)));
                float _cvt_f32_f16_35;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_35) : "h"((uint16_t)(_e4m3x2_to_f16x2_17 >> 16)));
                float _mul_22;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_34), "f"(rs_s_2));
                float _mul_23;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_35), "f"(rs_s_2));
                float2 scale0_558 = make_float2(_mul_22, _mul_22);
                float2 scale1_559 = make_float2(_mul_23, _mul_23);
                float _unpack_fp4_44[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[44 + 0]));
                    (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(_fp4_wire_values_0[44 + 0]));
                    #endif
                }
                float2 a_560 = make_float2(acc_407[0], acc_407[1]);
                float2 q_561 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                float2 r_562 = fma_f32x2_rn_noftz(q_561, scale0_558, a_560);
                acc_407[0] = r_562.x;
                acc_407[1] = r_562.y;
                float2 a_563 = make_float2(acc_407[2], acc_407[3]);
                float2 q_564 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                float2 r_565 = fma_f32x2_rn_noftz(q_564, scale0_558, a_563);
                acc_407[2] = r_565.x;
                acc_407[3] = r_565.y;
                float2 a_566 = make_float2(acc_407[4], acc_407[5]);
                float2 q_567 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                float2 r_568 = fma_f32x2_rn_noftz(q_567, scale0_558, a_566);
                acc_407[4] = r_568.x;
                acc_407[5] = r_568.y;
                float2 a_569 = make_float2(acc_407[6], acc_407[7]);
                float2 q_570 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                float2 r_571 = fma_f32x2_rn_noftz(q_570, scale0_558, a_569);
                acc_407[6] = r_571.x;
                acc_407[7] = r_571.y;
                float _unpack_fp4_45[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[45 + 0]));
                    (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(_fp4_wire_values_0[45 + 0]));
                    #endif
                }
                float2 a_572 = make_float2(acc_407[8], acc_407[9]);
                float2 q_573 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                float2 r_574 = fma_f32x2_rn_noftz(q_573, scale0_558, a_572);
                acc_407[8] = r_574.x;
                acc_407[9] = r_574.y;
                float2 a_575 = make_float2(acc_407[10], acc_407[11]);
                float2 q_576 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                float2 r_577 = fma_f32x2_rn_noftz(q_576, scale0_558, a_575);
                acc_407[10] = r_577.x;
                acc_407[11] = r_577.y;
                float2 a_578 = make_float2(acc_407[12], acc_407[13]);
                float2 q_579 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                float2 r_580 = fma_f32x2_rn_noftz(q_579, scale0_558, a_578);
                acc_407[12] = r_580.x;
                acc_407[13] = r_580.y;
                float2 a_581 = make_float2(acc_407[14], acc_407[15]);
                float2 q_582 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                float2 r_583 = fma_f32x2_rn_noftz(q_582, scale0_558, a_581);
                acc_407[14] = r_583.x;
                acc_407[15] = r_583.y;
                float _unpack_fp4_46[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[46 + 0]));
                    (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(_fp4_wire_values_0[46 + 0]));
                    #endif
                }
                float2 a_584 = make_float2(acc_407[16], acc_407[17]);
                float2 q_585 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                float2 r_586 = fma_f32x2_rn_noftz(q_585, scale1_559, a_584);
                acc_407[16] = r_586.x;
                acc_407[17] = r_586.y;
                float2 a_587 = make_float2(acc_407[18], acc_407[19]);
                float2 q_588 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                float2 r_589 = fma_f32x2_rn_noftz(q_588, scale1_559, a_587);
                acc_407[18] = r_589.x;
                acc_407[19] = r_589.y;
                float2 a_590 = make_float2(acc_407[20], acc_407[21]);
                float2 q_591 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                float2 r_592 = fma_f32x2_rn_noftz(q_591, scale1_559, a_590);
                acc_407[20] = r_592.x;
                acc_407[21] = r_592.y;
                float2 a_593 = make_float2(acc_407[22], acc_407[23]);
                float2 q_594 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                float2 r_595 = fma_f32x2_rn_noftz(q_594, scale1_559, a_593);
                acc_407[22] = r_595.x;
                acc_407[23] = r_595.y;
                float _unpack_fp4_47[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_0[47 + 0]));
                    (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(_fp4_wire_values_0[47 + 0]));
                    #endif
                }
                float2 a_596 = make_float2(acc_407[24], acc_407[25]);
                float2 q_597 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                float2 r_598 = fma_f32x2_rn_noftz(q_597, scale1_559, a_596);
                acc_407[24] = r_598.x;
                acc_407[25] = r_598.y;
                float2 a_599 = make_float2(acc_407[26], acc_407[27]);
                float2 q_600 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                float2 r_601 = fma_f32x2_rn_noftz(q_600, scale1_559, a_599);
                acc_407[26] = r_601.x;
                acc_407[27] = r_601.y;
                float2 a_602 = make_float2(acc_407[28], acc_407[29]);
                float2 q_603 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                float2 r_604 = fma_f32x2_rn_noftz(q_603, scale1_559, a_602);
                acc_407[28] = r_604.x;
                acc_407[29] = r_604.y;
                float2 a_605 = make_float2(acc_407[30], acc_407[31]);
                float2 q_606 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                float2 r_607 = fma_f32x2_rn_noftz(q_606, scale1_559, a_605);
                acc_407[30] = r_607.x;
                acc_407[31] = r_607.y;
                float am_1 = 0.0f;
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
                float cur_608 = am_1;
                cur_608 = fmaxf(cur_608, fabsf(acc_206[0]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[1]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[2]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[3]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[4]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[5]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[6]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[7]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[8]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[9]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[10]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[11]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[12]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[13]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[14]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[15]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[16]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[17]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[18]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[19]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[20]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[21]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[22]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[23]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[24]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[25]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[26]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[27]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[28]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[29]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[30]));
                cur_608 = fmaxf(cur_608, fabsf(acc_206[31]));
                am_1 = cur_608;
                float cur_609 = am_1;
                cur_609 = fmaxf(cur_609, fabsf(acc_407[0]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[1]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[2]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[3]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[4]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[5]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[6]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[7]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[8]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[9]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[10]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[11]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[12]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[13]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[14]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[15]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[16]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[17]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[18]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[19]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[20]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[21]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[22]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[23]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[24]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[25]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[26]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[27]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[28]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[29]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[30]));
                cur_609 = fmaxf(cur_609, fabsf(acc_407[31]));
                am_1 = cur_609;
                float _warp_reduce_144 = am_1;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_144 = max_noftz(_warp_reduce_144, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_144, offset));
                am_1 = _warp_reduce_144;
                int sam_base = 4 * t + 2 * kbp;
                if (lane_1 == 0) {
                    s_am[sam_base + h] = am_1;
                }
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                float am2 = fmaxf(s_am[sam_base], s_am[sam_base + 1]);
                float rs2 = 1.0f;
                float _fdiv_rn_13 = __fdiv_rn(2688.0f, am2);
                float inv_1 = 0.0f;
                if (am2 > 0.0f) {
                    inv_1 = _fdiv_rn_13;
                }
                float rs_1 = __fdiv_rn(am2, 2688.0f);
                rs2 = rs_1;
                int pp2_1 = lane_1 + 32 * (3 * h);
                unsigned int q_610[4];
                float bm0_1 = 0.0f;
                float bm1_1 = 0.0f;
                bm0_1 = fmaxf(bm0_1, fabsf(acc[0]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[16]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[1]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[17]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[2]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[18]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[3]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[19]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[4]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[20]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[5]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[21]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[6]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[22]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[7]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[23]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[8]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[24]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[9]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[25]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[10]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[26]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[11]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[27]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[12]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[28]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[13]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[29]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[14]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[30]));
                bm0_1 = fmaxf(bm0_1, fabsf(acc[15]));
                bm1_1 = fmaxf(bm1_1, fabsf(acc[31]));
                float _fdiv_145 = __fdiv_rn(mul_f32_rn_noftz(bm1_1, inv_1), 6.0f);
                float _fdiv_146 = __fdiv_rn(mul_f32_rn_noftz(bm0_1, inv_1), 6.0f);
                uint16_t _e4m3x2_f32_6;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(_fdiv_145), "f"(_fdiv_146));
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
                float2 pair_1 = make_float2(acc[0], acc[1]);
                float2 g_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm_1));
                uint32_t _fp4_pair_72;
                uint16_t _e2m1_pair_h_147;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_147) : "f"(g_1.x), "f"(g_1.y));
                _fp4_pair_72 = (uint32_t)_e2m1_pair_h_147;
                float2 pair_611 = make_float2(acc[2], acc[3]);
                float2 g_612;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_612) : "l"(*(const unsigned long long*)&pair_611), "l"(*(const unsigned long long*)&mm_1));
                uint32_t _fp4_pair_73;
                uint16_t _e2m1_pair_h_148;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_148) : "f"(g_612.x), "f"(g_612.y));
                _fp4_pair_73 = (uint32_t)_e2m1_pair_h_148;
                float2 pair_613 = make_float2(acc[4], acc[5]);
                float2 g_614;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_614) : "l"(*(const unsigned long long*)&pair_613), "l"(*(const unsigned long long*)&mm_1));
                uint32_t _fp4_pair_74;
                uint16_t _e2m1_pair_h_149;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_149) : "f"(g_614.x), "f"(g_614.y));
                _fp4_pair_74 = (uint32_t)_e2m1_pair_h_149;
                float2 pair_615 = make_float2(acc[6], acc[7]);
                float2 g_616;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_616) : "l"(*(const unsigned long long*)&pair_615), "l"(*(const unsigned long long*)&mm_1));
                uint32_t _fp4_e2m1_pack_pair_150;
                uint16_t _e2m1_pair_h_151;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_151) : "f"(g_616.x), "f"(g_616.y));
                _fp4_e2m1_pack_pair_150 = (uint32_t)_e2m1_pair_h_151;
                unsigned int word_1 = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_e2m1_pack_pair_150 << 24;
                unsigned int carry_1 = (word_1 & 2004318071) + 2004318071;
                q_610[0] = word_1 & (carry_1 | 2004318071);
                float2 mm_617 = make_float2(m0_1, m0_1);
                float2 pair_618 = make_float2(acc[8], acc[9]);
                float2 g_619;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_619) : "l"(*(const unsigned long long*)&pair_618), "l"(*(const unsigned long long*)&mm_617));
                uint32_t _fp4_pair_75;
                uint16_t _e2m1_pair_h_152;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_152) : "f"(g_619.x), "f"(g_619.y));
                _fp4_pair_75 = (uint32_t)_e2m1_pair_h_152;
                float2 pair_620 = make_float2(acc[10], acc[11]);
                float2 g_621;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_621) : "l"(*(const unsigned long long*)&pair_620), "l"(*(const unsigned long long*)&mm_617));
                uint32_t _fp4_pair_76;
                uint16_t _e2m1_pair_h_153;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_153) : "f"(g_621.x), "f"(g_621.y));
                _fp4_pair_76 = (uint32_t)_e2m1_pair_h_153;
                float2 pair_622 = make_float2(acc[12], acc[13]);
                float2 g_623;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_623) : "l"(*(const unsigned long long*)&pair_622), "l"(*(const unsigned long long*)&mm_617));
                uint32_t _fp4_pair_77;
                uint16_t _e2m1_pair_h_154;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_154) : "f"(g_623.x), "f"(g_623.y));
                _fp4_pair_77 = (uint32_t)_e2m1_pair_h_154;
                float2 pair_624 = make_float2(acc[14], acc[15]);
                float2 g_625;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_625) : "l"(*(const unsigned long long*)&pair_624), "l"(*(const unsigned long long*)&mm_617));
                uint32_t _fp4_e2m1_pack_pair_155;
                uint16_t _e2m1_pair_h_156;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_156) : "f"(g_625.x), "f"(g_625.y));
                _fp4_e2m1_pack_pair_155 = (uint32_t)_e2m1_pair_h_156;
                unsigned int word_626 = _fp4_pair_75 | _fp4_pair_76 << 8 | _fp4_pair_77 << 16 | _fp4_e2m1_pack_pair_155 << 24;
                unsigned int carry_627 = (word_626 & 2004318071) + 2004318071;
                q_610[1] = word_626 & (carry_627 | 2004318071);
                float2 mm_628 = make_float2(m1_1, m1_1);
                float2 pair_629 = make_float2(acc[16], acc[17]);
                float2 g_630;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_630) : "l"(*(const unsigned long long*)&pair_629), "l"(*(const unsigned long long*)&mm_628));
                uint32_t _fp4_pair_78;
                uint16_t _e2m1_pair_h_157;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_157) : "f"(g_630.x), "f"(g_630.y));
                _fp4_pair_78 = (uint32_t)_e2m1_pair_h_157;
                float2 pair_631 = make_float2(acc[18], acc[19]);
                float2 g_632;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_632) : "l"(*(const unsigned long long*)&pair_631), "l"(*(const unsigned long long*)&mm_628));
                uint32_t _fp4_pair_79;
                uint16_t _e2m1_pair_h_158;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_158) : "f"(g_632.x), "f"(g_632.y));
                _fp4_pair_79 = (uint32_t)_e2m1_pair_h_158;
                float2 pair_633 = make_float2(acc[20], acc[21]);
                float2 g_634;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_634) : "l"(*(const unsigned long long*)&pair_633), "l"(*(const unsigned long long*)&mm_628));
                uint32_t _fp4_pair_80;
                uint16_t _e2m1_pair_h_159;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_159) : "f"(g_634.x), "f"(g_634.y));
                _fp4_pair_80 = (uint32_t)_e2m1_pair_h_159;
                float2 pair_635 = make_float2(acc[22], acc[23]);
                float2 g_636;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_636) : "l"(*(const unsigned long long*)&pair_635), "l"(*(const unsigned long long*)&mm_628));
                uint32_t _fp4_e2m1_pack_pair_160;
                uint16_t _e2m1_pair_h_161;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_161) : "f"(g_636.x), "f"(g_636.y));
                _fp4_e2m1_pack_pair_160 = (uint32_t)_e2m1_pair_h_161;
                unsigned int word_637 = _fp4_pair_78 | _fp4_pair_79 << 8 | _fp4_pair_80 << 16 | _fp4_e2m1_pack_pair_160 << 24;
                unsigned int carry_638 = (word_637 & 2004318071) + 2004318071;
                q_610[2] = word_637 & (carry_638 | 2004318071);
                float2 mm_639 = make_float2(m1_1, m1_1);
                float2 pair_640 = make_float2(acc[24], acc[25]);
                float2 g_641;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_641) : "l"(*(const unsigned long long*)&pair_640), "l"(*(const unsigned long long*)&mm_639));
                uint32_t _fp4_pair_81;
                uint16_t _e2m1_pair_h_162;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_162) : "f"(g_641.x), "f"(g_641.y));
                _fp4_pair_81 = (uint32_t)_e2m1_pair_h_162;
                float2 pair_642 = make_float2(acc[26], acc[27]);
                float2 g_643;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_643) : "l"(*(const unsigned long long*)&pair_642), "l"(*(const unsigned long long*)&mm_639));
                uint32_t _fp4_pair_82;
                uint16_t _e2m1_pair_h_163;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_163) : "f"(g_643.x), "f"(g_643.y));
                _fp4_pair_82 = (uint32_t)_e2m1_pair_h_163;
                float2 pair_644 = make_float2(acc[28], acc[29]);
                float2 g_645;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_645) : "l"(*(const unsigned long long*)&pair_644), "l"(*(const unsigned long long*)&mm_639));
                uint32_t _fp4_pair_83;
                uint16_t _e2m1_pair_h_164;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_164) : "f"(g_645.x), "f"(g_645.y));
                _fp4_pair_83 = (uint32_t)_e2m1_pair_h_164;
                float2 pair_646 = make_float2(acc[30], acc[31]);
                float2 g_647;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_647) : "l"(*(const unsigned long long*)&pair_646), "l"(*(const unsigned long long*)&mm_639));
                uint32_t _fp4_e2m1_pack_pair_165;
                uint16_t _e2m1_pair_h_166;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_166) : "f"(g_647.x), "f"(g_647.y));
                _fp4_e2m1_pack_pair_165 = (uint32_t)_e2m1_pair_h_166;
                unsigned int word_648 = _fp4_pair_81 | _fp4_pair_82 << 8 | _fp4_pair_83 << 16 | _fp4_e2m1_pack_pair_165 << 24;
                unsigned int carry_649 = (word_648 & 2004318071) + 2004318071;
                q_610[3] = word_648 & (carry_649 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_1)), "r"(q_610[0]), "r"(q_610[1]), "r"(q_610[2]), "r"(q_610[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_1)), "h"((uint16_t)(_e4m3x2_f32_6)) : "memory");
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
                float2 scale0_650 = make_float2(_mul_24, _mul_24);
                float2 scale1_651 = make_float2(_mul_25, _mul_25);
                float _unpack_fp4_48[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_610[0]));
                    (&_unpack_fp4_48[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_48[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_48[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_48[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_48[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_48[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_48[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_48[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_48[0]), "=f"(_unpack_fp4_48[1]), "=f"(_unpack_fp4_48[2]), "=f"(_unpack_fp4_48[3]), "=f"(_unpack_fp4_48[4]), "=f"(_unpack_fp4_48[5]), "=f"(_unpack_fp4_48[6]), "=f"(_unpack_fp4_48[7]) : "r"(q_610[0]));
                    #endif
                }
                float2 a_652 = make_float2(d[0], d[1]);
                float2 q_653 = make_float2(_unpack_fp4_48[0], _unpack_fp4_48[1]);
                float2 r_654 = fma_f32x2_rn_noftz(q_653, scale0_650, a_652);
                d[0] = r_654.x;
                d[1] = r_654.y;
                float2 a_655 = make_float2(d[2], d[3]);
                float2 q_656 = make_float2(_unpack_fp4_48[2], _unpack_fp4_48[3]);
                float2 r_657 = fma_f32x2_rn_noftz(q_656, scale0_650, a_655);
                d[2] = r_657.x;
                d[3] = r_657.y;
                float2 a_658 = make_float2(d[4], d[5]);
                float2 q_659 = make_float2(_unpack_fp4_48[4], _unpack_fp4_48[5]);
                float2 r_660 = fma_f32x2_rn_noftz(q_659, scale0_650, a_658);
                d[4] = r_660.x;
                d[5] = r_660.y;
                float2 a_661 = make_float2(d[6], d[7]);
                float2 q_662 = make_float2(_unpack_fp4_48[6], _unpack_fp4_48[7]);
                float2 r_663 = fma_f32x2_rn_noftz(q_662, scale0_650, a_661);
                d[6] = r_663.x;
                d[7] = r_663.y;
                float _unpack_fp4_49[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_610[1 + 0]));
                    (&_unpack_fp4_49[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_49[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_49[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_49[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_49[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_49[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_49[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_49[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_49[0]), "=f"(_unpack_fp4_49[1]), "=f"(_unpack_fp4_49[2]), "=f"(_unpack_fp4_49[3]), "=f"(_unpack_fp4_49[4]), "=f"(_unpack_fp4_49[5]), "=f"(_unpack_fp4_49[6]), "=f"(_unpack_fp4_49[7]) : "r"(q_610[1 + 0]));
                    #endif
                }
                float2 a_664 = make_float2(d[8], d[9]);
                float2 q_665 = make_float2(_unpack_fp4_49[0], _unpack_fp4_49[1]);
                float2 r_666 = fma_f32x2_rn_noftz(q_665, scale0_650, a_664);
                d[8] = r_666.x;
                d[9] = r_666.y;
                float2 a_667 = make_float2(d[10], d[11]);
                float2 q_668 = make_float2(_unpack_fp4_49[2], _unpack_fp4_49[3]);
                float2 r_669 = fma_f32x2_rn_noftz(q_668, scale0_650, a_667);
                d[10] = r_669.x;
                d[11] = r_669.y;
                float2 a_670 = make_float2(d[12], d[13]);
                float2 q_671 = make_float2(_unpack_fp4_49[4], _unpack_fp4_49[5]);
                float2 r_672 = fma_f32x2_rn_noftz(q_671, scale0_650, a_670);
                d[12] = r_672.x;
                d[13] = r_672.y;
                float2 a_673 = make_float2(d[14], d[15]);
                float2 q_674 = make_float2(_unpack_fp4_49[6], _unpack_fp4_49[7]);
                float2 r_675 = fma_f32x2_rn_noftz(q_674, scale0_650, a_673);
                d[14] = r_675.x;
                d[15] = r_675.y;
                float _unpack_fp4_50[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_610[2 + 0]));
                    (&_unpack_fp4_50[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_50[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_50[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_50[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_50[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_50[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_50[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_50[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_50[0]), "=f"(_unpack_fp4_50[1]), "=f"(_unpack_fp4_50[2]), "=f"(_unpack_fp4_50[3]), "=f"(_unpack_fp4_50[4]), "=f"(_unpack_fp4_50[5]), "=f"(_unpack_fp4_50[6]), "=f"(_unpack_fp4_50[7]) : "r"(q_610[2 + 0]));
                    #endif
                }
                float2 a_676 = make_float2(d[16], d[17]);
                float2 q_677 = make_float2(_unpack_fp4_50[0], _unpack_fp4_50[1]);
                float2 r_678 = fma_f32x2_rn_noftz(q_677, scale1_651, a_676);
                d[16] = r_678.x;
                d[17] = r_678.y;
                float2 a_679 = make_float2(d[18], d[19]);
                float2 q_680 = make_float2(_unpack_fp4_50[2], _unpack_fp4_50[3]);
                float2 r_681 = fma_f32x2_rn_noftz(q_680, scale1_651, a_679);
                d[18] = r_681.x;
                d[19] = r_681.y;
                float2 a_682 = make_float2(d[20], d[21]);
                float2 q_683 = make_float2(_unpack_fp4_50[4], _unpack_fp4_50[5]);
                float2 r_684 = fma_f32x2_rn_noftz(q_683, scale1_651, a_682);
                d[20] = r_684.x;
                d[21] = r_684.y;
                float2 a_685 = make_float2(d[22], d[23]);
                float2 q_686 = make_float2(_unpack_fp4_50[6], _unpack_fp4_50[7]);
                float2 r_687 = fma_f32x2_rn_noftz(q_686, scale1_651, a_685);
                d[22] = r_687.x;
                d[23] = r_687.y;
                float _unpack_fp4_51[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_610[3 + 0]));
                    (&_unpack_fp4_51[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_51[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_51[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_51[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_51[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_51[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_51[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_51[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_51[0]), "=f"(_unpack_fp4_51[1]), "=f"(_unpack_fp4_51[2]), "=f"(_unpack_fp4_51[3]), "=f"(_unpack_fp4_51[4]), "=f"(_unpack_fp4_51[5]), "=f"(_unpack_fp4_51[6]), "=f"(_unpack_fp4_51[7]) : "r"(q_610[3 + 0]));
                    #endif
                }
                float2 a_688 = make_float2(d[24], d[25]);
                float2 q_689 = make_float2(_unpack_fp4_51[0], _unpack_fp4_51[1]);
                float2 r_690 = fma_f32x2_rn_noftz(q_689, scale1_651, a_688);
                d[24] = r_690.x;
                d[25] = r_690.y;
                float2 a_691 = make_float2(d[26], d[27]);
                float2 q_692 = make_float2(_unpack_fp4_51[2], _unpack_fp4_51[3]);
                float2 r_693 = fma_f32x2_rn_noftz(q_692, scale1_651, a_691);
                d[26] = r_693.x;
                d[27] = r_693.y;
                float2 a_694 = make_float2(d[28], d[29]);
                float2 q_695 = make_float2(_unpack_fp4_51[4], _unpack_fp4_51[5]);
                float2 r_696 = fma_f32x2_rn_noftz(q_695, scale1_651, a_694);
                d[28] = r_696.x;
                d[29] = r_696.y;
                float2 a_697 = make_float2(d[30], d[31]);
                float2 q_698 = make_float2(_unpack_fp4_51[6], _unpack_fp4_51[7]);
                float2 r_699 = fma_f32x2_rn_noftz(q_698, scale1_651, a_697);
                d[30] = r_699.x;
                d[31] = r_699.y;
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
                int c = lane_1;
                int l = c >> 2;
                unsigned int w4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l + (c & 3 ^ l >> 1 & 3)))));
                long long idx = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c));
                {
                    int4 _iv4 = make_int4(w4[0 + 0], w4[0 + 1], w4[0 + 2], w4[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                }
                int c_700 = lane_1 + 32;
                int l_701 = c_700 >> 2;
                unsigned int w4_702[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_702[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_702[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_702[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_702[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_701 + (c_700 & 3 ^ l_701 >> 1 & 3)))));
                long long idx_703 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_700));
                {
                    int4 _iv4 = make_int4(w4_702[0 + 0], w4_702[0 + 1], w4_702[0 + 2], w4_702[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_703) = _iv4;
                }
                int c_704 = lane_1 + 64;
                int l_705 = c_704 >> 2;
                unsigned int w4_706[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_706[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_706[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_706[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_706[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_705 + (c_704 & 3 ^ l_705 >> 1 & 3)))));
                long long idx_707 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_704));
                {
                    int4 _iv4 = make_int4(w4_706[0 + 0], w4_706[0 + 1], w4_706[0 + 2], w4_706[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_707) = _iv4;
                }
                int c_708 = lane_1 + 96;
                int l_709 = c_708 >> 2;
                unsigned int w4_710[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_710[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_710[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_710[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_710[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_709 + (c_708 & 3 ^ l_709 >> 1 & 3)))));
                long long idx_711 = trow64 * 3072 + (long long)(4 * (128 * (3 * h) + c_708));
                {
                    int4 _iv4 = make_int4(w4_710[0 + 0], w4_710[0 + 1], w4_710[0 + 2], w4_710[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_711) = _iv4;
                }
                __syncwarp();
                int pp2_712 = lane_1 + 32 * (3 * h + 1);
                unsigned int q_713[4];
                float bm0_714 = 0.0f;
                float bm1_715 = 0.0f;
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[0]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[16]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[1]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[17]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[2]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[18]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[3]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[19]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[4]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[20]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[5]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[21]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[6]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[22]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[7]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[23]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[8]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[24]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[9]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[25]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[10]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[26]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[11]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[27]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[12]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[28]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[13]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[29]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[14]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[30]));
                bm0_714 = fmaxf(bm0_714, fabsf(acc_206[15]));
                bm1_715 = fmaxf(bm1_715, fabsf(acc_206[31]));
                float _fdiv_167 = __fdiv_rn(mul_f32_rn_noftz(bm1_715, inv_1), 6.0f);
                float _fdiv_168 = __fdiv_rn(mul_f32_rn_noftz(bm0_714, inv_1), 6.0f);
                uint16_t _e4m3x2_f32_7;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(_fdiv_167), "f"(_fdiv_168));
                uint32_t _e4m3x2_to_f16x2_20;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_20) : "h"((uint16_t)(_e4m3x2_f32_7)));
                float _cvt_f32_f16_40;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_40) : "h"((uint16_t)(_e4m3x2_to_f16x2_20)));
                float _cvt_f32_f16_41;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_41) : "h"((uint16_t)(_e4m3x2_to_f16x2_20 >> 16)));
                float _fdiv_rn_16 = __fdiv_rn(inv_1, _cvt_f32_f16_40);
                float _fdiv_rn_17 = __fdiv_rn(inv_1, _cvt_f32_f16_41);
                float m0_716 = 0.0f;
                float m1_717 = 0.0f;
                if (_cvt_f32_f16_40 > 0.0f) {
                    m0_716 = _fdiv_rn_16;
                }
                if (_cvt_f32_f16_41 > 0.0f) {
                    m1_717 = _fdiv_rn_17;
                }
                float2 mm_718 = make_float2(m0_716, m0_716);
                float2 pair_719 = make_float2(acc_206[0], acc_206[1]);
                float2 g_720;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_720) : "l"(*(const unsigned long long*)&pair_719), "l"(*(const unsigned long long*)&mm_718));
                uint32_t _fp4_pair_84;
                uint16_t _e2m1_pair_h_169;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_169) : "f"(g_720.x), "f"(g_720.y));
                _fp4_pair_84 = (uint32_t)_e2m1_pair_h_169;
                float2 pair_721 = make_float2(acc_206[2], acc_206[3]);
                float2 g_722;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_722) : "l"(*(const unsigned long long*)&pair_721), "l"(*(const unsigned long long*)&mm_718));
                uint32_t _fp4_pair_85;
                uint16_t _e2m1_pair_h_170;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_170) : "f"(g_722.x), "f"(g_722.y));
                _fp4_pair_85 = (uint32_t)_e2m1_pair_h_170;
                float2 pair_723 = make_float2(acc_206[4], acc_206[5]);
                float2 g_724;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_724) : "l"(*(const unsigned long long*)&pair_723), "l"(*(const unsigned long long*)&mm_718));
                uint32_t _fp4_pair_86;
                uint16_t _e2m1_pair_h_171;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_171) : "f"(g_724.x), "f"(g_724.y));
                _fp4_pair_86 = (uint32_t)_e2m1_pair_h_171;
                float2 pair_725 = make_float2(acc_206[6], acc_206[7]);
                float2 g_726;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_726) : "l"(*(const unsigned long long*)&pair_725), "l"(*(const unsigned long long*)&mm_718));
                uint32_t _fp4_e2m1_pack_pair_172;
                uint16_t _e2m1_pair_h_173;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_173) : "f"(g_726.x), "f"(g_726.y));
                _fp4_e2m1_pack_pair_172 = (uint32_t)_e2m1_pair_h_173;
                unsigned int word_727 = _fp4_pair_84 | _fp4_pair_85 << 8 | _fp4_pair_86 << 16 | _fp4_e2m1_pack_pair_172 << 24;
                unsigned int carry_728 = (word_727 & 2004318071) + 2004318071;
                q_713[0] = word_727 & (carry_728 | 2004318071);
                float2 mm_729 = make_float2(m0_716, m0_716);
                float2 pair_730 = make_float2(acc_206[8], acc_206[9]);
                float2 g_731;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_731) : "l"(*(const unsigned long long*)&pair_730), "l"(*(const unsigned long long*)&mm_729));
                uint32_t _fp4_pair_87;
                uint16_t _e2m1_pair_h_174;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_174) : "f"(g_731.x), "f"(g_731.y));
                _fp4_pair_87 = (uint32_t)_e2m1_pair_h_174;
                float2 pair_732 = make_float2(acc_206[10], acc_206[11]);
                float2 g_733;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_733) : "l"(*(const unsigned long long*)&pair_732), "l"(*(const unsigned long long*)&mm_729));
                uint32_t _fp4_pair_88;
                uint16_t _e2m1_pair_h_175;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_175) : "f"(g_733.x), "f"(g_733.y));
                _fp4_pair_88 = (uint32_t)_e2m1_pair_h_175;
                float2 pair_734 = make_float2(acc_206[12], acc_206[13]);
                float2 g_735;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_735) : "l"(*(const unsigned long long*)&pair_734), "l"(*(const unsigned long long*)&mm_729));
                uint32_t _fp4_pair_89;
                uint16_t _e2m1_pair_h_176;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_176) : "f"(g_735.x), "f"(g_735.y));
                _fp4_pair_89 = (uint32_t)_e2m1_pair_h_176;
                float2 pair_736 = make_float2(acc_206[14], acc_206[15]);
                float2 g_737;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_737) : "l"(*(const unsigned long long*)&pair_736), "l"(*(const unsigned long long*)&mm_729));
                uint32_t _fp4_e2m1_pack_pair_177;
                uint16_t _e2m1_pair_h_178;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_178) : "f"(g_737.x), "f"(g_737.y));
                _fp4_e2m1_pack_pair_177 = (uint32_t)_e2m1_pair_h_178;
                unsigned int word_738 = _fp4_pair_87 | _fp4_pair_88 << 8 | _fp4_pair_89 << 16 | _fp4_e2m1_pack_pair_177 << 24;
                unsigned int carry_739 = (word_738 & 2004318071) + 2004318071;
                q_713[1] = word_738 & (carry_739 | 2004318071);
                float2 mm_740 = make_float2(m1_717, m1_717);
                float2 pair_741 = make_float2(acc_206[16], acc_206[17]);
                float2 g_742;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_742) : "l"(*(const unsigned long long*)&pair_741), "l"(*(const unsigned long long*)&mm_740));
                uint32_t _fp4_pair_90;
                uint16_t _e2m1_pair_h_179;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_179) : "f"(g_742.x), "f"(g_742.y));
                _fp4_pair_90 = (uint32_t)_e2m1_pair_h_179;
                float2 pair_743 = make_float2(acc_206[18], acc_206[19]);
                float2 g_744;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_744) : "l"(*(const unsigned long long*)&pair_743), "l"(*(const unsigned long long*)&mm_740));
                uint32_t _fp4_pair_91;
                uint16_t _e2m1_pair_h_180;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_180) : "f"(g_744.x), "f"(g_744.y));
                _fp4_pair_91 = (uint32_t)_e2m1_pair_h_180;
                float2 pair_745 = make_float2(acc_206[20], acc_206[21]);
                float2 g_746;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_746) : "l"(*(const unsigned long long*)&pair_745), "l"(*(const unsigned long long*)&mm_740));
                uint32_t _fp4_pair_92;
                uint16_t _e2m1_pair_h_181;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_181) : "f"(g_746.x), "f"(g_746.y));
                _fp4_pair_92 = (uint32_t)_e2m1_pair_h_181;
                float2 pair_747 = make_float2(acc_206[22], acc_206[23]);
                float2 g_748;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_748) : "l"(*(const unsigned long long*)&pair_747), "l"(*(const unsigned long long*)&mm_740));
                uint32_t _fp4_e2m1_pack_pair_182;
                uint16_t _e2m1_pair_h_183;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_183) : "f"(g_748.x), "f"(g_748.y));
                _fp4_e2m1_pack_pair_182 = (uint32_t)_e2m1_pair_h_183;
                unsigned int word_749 = _fp4_pair_90 | _fp4_pair_91 << 8 | _fp4_pair_92 << 16 | _fp4_e2m1_pack_pair_182 << 24;
                unsigned int carry_750 = (word_749 & 2004318071) + 2004318071;
                q_713[2] = word_749 & (carry_750 | 2004318071);
                float2 mm_751 = make_float2(m1_717, m1_717);
                float2 pair_752 = make_float2(acc_206[24], acc_206[25]);
                float2 g_753;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_753) : "l"(*(const unsigned long long*)&pair_752), "l"(*(const unsigned long long*)&mm_751));
                uint32_t _fp4_pair_93;
                uint16_t _e2m1_pair_h_184;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_184) : "f"(g_753.x), "f"(g_753.y));
                _fp4_pair_93 = (uint32_t)_e2m1_pair_h_184;
                float2 pair_754 = make_float2(acc_206[26], acc_206[27]);
                float2 g_755;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_755) : "l"(*(const unsigned long long*)&pair_754), "l"(*(const unsigned long long*)&mm_751));
                uint32_t _fp4_pair_94;
                uint16_t _e2m1_pair_h_185;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_185) : "f"(g_755.x), "f"(g_755.y));
                _fp4_pair_94 = (uint32_t)_e2m1_pair_h_185;
                float2 pair_756 = make_float2(acc_206[28], acc_206[29]);
                float2 g_757;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_757) : "l"(*(const unsigned long long*)&pair_756), "l"(*(const unsigned long long*)&mm_751));
                uint32_t _fp4_pair_95;
                uint16_t _e2m1_pair_h_186;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_186) : "f"(g_757.x), "f"(g_757.y));
                _fp4_pair_95 = (uint32_t)_e2m1_pair_h_186;
                float2 pair_758 = make_float2(acc_206[30], acc_206[31]);
                float2 g_759;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_759) : "l"(*(const unsigned long long*)&pair_758), "l"(*(const unsigned long long*)&mm_751));
                uint32_t _fp4_e2m1_pack_pair_187;
                uint16_t _e2m1_pair_h_188;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_188) : "f"(g_759.x), "f"(g_759.y));
                _fp4_e2m1_pack_pair_187 = (uint32_t)_e2m1_pair_h_188;
                unsigned int word_760 = _fp4_pair_93 | _fp4_pair_94 << 8 | _fp4_pair_95 << 16 | _fp4_e2m1_pack_pair_187 << 24;
                unsigned int carry_761 = (word_760 & 2004318071) + 2004318071;
                q_713[3] = word_760 & (carry_761 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_712)), "r"(q_713[0]), "r"(q_713[1]), "r"(q_713[2]), "r"(q_713[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_712)), "h"((uint16_t)(_e4m3x2_f32_7)) : "memory");
                float d_762[32] = {0};
                uint32_t _e4m3x2_to_f16x2_21;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_21) : "h"((uint16_t)(_e4m3x2_f32_7)));
                float _cvt_f32_f16_42;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_42) : "h"((uint16_t)(_e4m3x2_to_f16x2_21)));
                float _cvt_f32_f16_43;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_43) : "h"((uint16_t)(_e4m3x2_to_f16x2_21 >> 16)));
                float _mul_26;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_cvt_f32_f16_42), "f"(rs2));
                float _mul_27;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_cvt_f32_f16_43), "f"(rs2));
                float2 scale0_763 = make_float2(_mul_26, _mul_26);
                float2 scale1_764 = make_float2(_mul_27, _mul_27);
                float _unpack_fp4_52[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_713[0]));
                    (&_unpack_fp4_52[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_52[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_52[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_52[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_52[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_52[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_52[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_52[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_52[0]), "=f"(_unpack_fp4_52[1]), "=f"(_unpack_fp4_52[2]), "=f"(_unpack_fp4_52[3]), "=f"(_unpack_fp4_52[4]), "=f"(_unpack_fp4_52[5]), "=f"(_unpack_fp4_52[6]), "=f"(_unpack_fp4_52[7]) : "r"(q_713[0]));
                    #endif
                }
                float2 a_765 = make_float2(d_762[0], d_762[1]);
                float2 q_766 = make_float2(_unpack_fp4_52[0], _unpack_fp4_52[1]);
                float2 r_767 = fma_f32x2_rn_noftz(q_766, scale0_763, a_765);
                d_762[0] = r_767.x;
                d_762[1] = r_767.y;
                float2 a_768 = make_float2(d_762[2], d_762[3]);
                float2 q_769 = make_float2(_unpack_fp4_52[2], _unpack_fp4_52[3]);
                float2 r_770 = fma_f32x2_rn_noftz(q_769, scale0_763, a_768);
                d_762[2] = r_770.x;
                d_762[3] = r_770.y;
                float2 a_771 = make_float2(d_762[4], d_762[5]);
                float2 q_772 = make_float2(_unpack_fp4_52[4], _unpack_fp4_52[5]);
                float2 r_773 = fma_f32x2_rn_noftz(q_772, scale0_763, a_771);
                d_762[4] = r_773.x;
                d_762[5] = r_773.y;
                float2 a_774 = make_float2(d_762[6], d_762[7]);
                float2 q_775 = make_float2(_unpack_fp4_52[6], _unpack_fp4_52[7]);
                float2 r_776 = fma_f32x2_rn_noftz(q_775, scale0_763, a_774);
                d_762[6] = r_776.x;
                d_762[7] = r_776.y;
                float _unpack_fp4_53[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_713[1 + 0]));
                    (&_unpack_fp4_53[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_53[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_53[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_53[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_53[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_53[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_53[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_53[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_53[0]), "=f"(_unpack_fp4_53[1]), "=f"(_unpack_fp4_53[2]), "=f"(_unpack_fp4_53[3]), "=f"(_unpack_fp4_53[4]), "=f"(_unpack_fp4_53[5]), "=f"(_unpack_fp4_53[6]), "=f"(_unpack_fp4_53[7]) : "r"(q_713[1 + 0]));
                    #endif
                }
                float2 a_777 = make_float2(d_762[8], d_762[9]);
                float2 q_778 = make_float2(_unpack_fp4_53[0], _unpack_fp4_53[1]);
                float2 r_779 = fma_f32x2_rn_noftz(q_778, scale0_763, a_777);
                d_762[8] = r_779.x;
                d_762[9] = r_779.y;
                float2 a_780 = make_float2(d_762[10], d_762[11]);
                float2 q_781 = make_float2(_unpack_fp4_53[2], _unpack_fp4_53[3]);
                float2 r_782 = fma_f32x2_rn_noftz(q_781, scale0_763, a_780);
                d_762[10] = r_782.x;
                d_762[11] = r_782.y;
                float2 a_783 = make_float2(d_762[12], d_762[13]);
                float2 q_784 = make_float2(_unpack_fp4_53[4], _unpack_fp4_53[5]);
                float2 r_785 = fma_f32x2_rn_noftz(q_784, scale0_763, a_783);
                d_762[12] = r_785.x;
                d_762[13] = r_785.y;
                float2 a_786 = make_float2(d_762[14], d_762[15]);
                float2 q_787 = make_float2(_unpack_fp4_53[6], _unpack_fp4_53[7]);
                float2 r_788 = fma_f32x2_rn_noftz(q_787, scale0_763, a_786);
                d_762[14] = r_788.x;
                d_762[15] = r_788.y;
                float _unpack_fp4_54[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_713[2 + 0]));
                    (&_unpack_fp4_54[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_54[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_54[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_54[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_54[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_54[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_54[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_54[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_54[0]), "=f"(_unpack_fp4_54[1]), "=f"(_unpack_fp4_54[2]), "=f"(_unpack_fp4_54[3]), "=f"(_unpack_fp4_54[4]), "=f"(_unpack_fp4_54[5]), "=f"(_unpack_fp4_54[6]), "=f"(_unpack_fp4_54[7]) : "r"(q_713[2 + 0]));
                    #endif
                }
                float2 a_789 = make_float2(d_762[16], d_762[17]);
                float2 q_790 = make_float2(_unpack_fp4_54[0], _unpack_fp4_54[1]);
                float2 r_791 = fma_f32x2_rn_noftz(q_790, scale1_764, a_789);
                d_762[16] = r_791.x;
                d_762[17] = r_791.y;
                float2 a_792 = make_float2(d_762[18], d_762[19]);
                float2 q_793 = make_float2(_unpack_fp4_54[2], _unpack_fp4_54[3]);
                float2 r_794 = fma_f32x2_rn_noftz(q_793, scale1_764, a_792);
                d_762[18] = r_794.x;
                d_762[19] = r_794.y;
                float2 a_795 = make_float2(d_762[20], d_762[21]);
                float2 q_796 = make_float2(_unpack_fp4_54[4], _unpack_fp4_54[5]);
                float2 r_797 = fma_f32x2_rn_noftz(q_796, scale1_764, a_795);
                d_762[20] = r_797.x;
                d_762[21] = r_797.y;
                float2 a_798 = make_float2(d_762[22], d_762[23]);
                float2 q_799 = make_float2(_unpack_fp4_54[6], _unpack_fp4_54[7]);
                float2 r_800 = fma_f32x2_rn_noftz(q_799, scale1_764, a_798);
                d_762[22] = r_800.x;
                d_762[23] = r_800.y;
                float _unpack_fp4_55[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_713[3 + 0]));
                    (&_unpack_fp4_55[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_55[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_55[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_55[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_55[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_55[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_55[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_55[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_55[0]), "=f"(_unpack_fp4_55[1]), "=f"(_unpack_fp4_55[2]), "=f"(_unpack_fp4_55[3]), "=f"(_unpack_fp4_55[4]), "=f"(_unpack_fp4_55[5]), "=f"(_unpack_fp4_55[6]), "=f"(_unpack_fp4_55[7]) : "r"(q_713[3 + 0]));
                    #endif
                }
                float2 a_801 = make_float2(d_762[24], d_762[25]);
                float2 q_802 = make_float2(_unpack_fp4_55[0], _unpack_fp4_55[1]);
                float2 r_803 = fma_f32x2_rn_noftz(q_802, scale1_764, a_801);
                d_762[24] = r_803.x;
                d_762[25] = r_803.y;
                float2 a_804 = make_float2(d_762[26], d_762[27]);
                float2 q_805 = make_float2(_unpack_fp4_55[2], _unpack_fp4_55[3]);
                float2 r_806 = fma_f32x2_rn_noftz(q_805, scale1_764, a_804);
                d_762[26] = r_806.x;
                d_762[27] = r_806.y;
                float2 a_807 = make_float2(d_762[28], d_762[29]);
                float2 q_808 = make_float2(_unpack_fp4_55[4], _unpack_fp4_55[5]);
                float2 r_809 = fma_f32x2_rn_noftz(q_808, scale1_764, a_807);
                d_762[28] = r_809.x;
                d_762[29] = r_809.y;
                float2 a_810 = make_float2(d_762[30], d_762[31]);
                float2 q_811 = make_float2(_unpack_fp4_55[6], _unpack_fp4_55[7]);
                float2 r_812 = fma_f32x2_rn_noftz(q_811, scale1_764, a_810);
                d_762[30] = r_812.x;
                d_762[31] = r_812.y;
                unsigned int d_762_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_762[_lp*2 + 0], d_762[_lp*2+1 + 0]));
                    d_762_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_762_bf16[0]), "r"(d_762_bf16[1]), "r"(d_762_bf16[2]), "r"(d_762_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_762_bf16[4]), "r"(d_762_bf16[5]), "r"(d_762_bf16[6]), "r"(d_762_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_762_bf16[8]), "r"(d_762_bf16[9]), "r"(d_762_bf16[10]), "r"(d_762_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_762_bf16[12]), "r"(d_762_bf16[13]), "r"(d_762_bf16[14]), "r"(d_762_bf16[15]) : "memory");
                __syncwarp();
                int c_813 = lane_1;
                int l_814 = c_813 >> 2;
                unsigned int w4_815[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_815[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_815[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_815[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_815[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_814 + (c_813 & 3 ^ l_814 >> 1 & 3)))));
                long long idx_816 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_813));
                {
                    int4 _iv4 = make_int4(w4_815[0 + 0], w4_815[0 + 1], w4_815[0 + 2], w4_815[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_816) = _iv4;
                }
                int c_817 = lane_1 + 32;
                int l_818 = c_817 >> 2;
                unsigned int w4_819[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_819[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_819[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_819[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_819[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_818 + (c_817 & 3 ^ l_818 >> 1 & 3)))));
                long long idx_820 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_817));
                {
                    int4 _iv4 = make_int4(w4_819[0 + 0], w4_819[0 + 1], w4_819[0 + 2], w4_819[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_820) = _iv4;
                }
                int c_821 = lane_1 + 64;
                int l_822 = c_821 >> 2;
                unsigned int w4_823[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_823[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_823[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_823[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_823[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_822 + (c_821 & 3 ^ l_822 >> 1 & 3)))));
                long long idx_824 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_821));
                {
                    int4 _iv4 = make_int4(w4_823[0 + 0], w4_823[0 + 1], w4_823[0 + 2], w4_823[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_824) = _iv4;
                }
                int c_825 = lane_1 + 96;
                int l_826 = c_825 >> 2;
                unsigned int w4_827[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_827[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_827[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_827[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_827[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_826 + (c_825 & 3 ^ l_826 >> 1 & 3)))));
                long long idx_828 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 1) + c_825));
                {
                    int4 _iv4 = make_int4(w4_827[0 + 0], w4_827[0 + 1], w4_827[0 + 2], w4_827[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_828) = _iv4;
                }
                __syncwarp();
                int pp2_829 = lane_1 + 32 * (3 * h + 2);
                unsigned int q_830[4];
                float bm0_831 = 0.0f;
                float bm1_832 = 0.0f;
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[0]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[16]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[1]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[17]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[2]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[18]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[3]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[19]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[4]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[20]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[5]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[21]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[6]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[22]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[7]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[23]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[8]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[24]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[9]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[25]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[10]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[26]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[11]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[27]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[12]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[28]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[13]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[29]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[14]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[30]));
                bm0_831 = fmaxf(bm0_831, fabsf(acc_407[15]));
                bm1_832 = fmaxf(bm1_832, fabsf(acc_407[31]));
                float _fdiv_189 = __fdiv_rn(mul_f32_rn_noftz(bm1_832, inv_1), 6.0f);
                float _fdiv_190 = __fdiv_rn(mul_f32_rn_noftz(bm0_831, inv_1), 6.0f);
                uint16_t _e4m3x2_f32_8;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_8) : "f"(_fdiv_189), "f"(_fdiv_190));
                uint32_t _e4m3x2_to_f16x2_22;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_22) : "h"((uint16_t)(_e4m3x2_f32_8)));
                float _cvt_f32_f16_44;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_44) : "h"((uint16_t)(_e4m3x2_to_f16x2_22)));
                float _cvt_f32_f16_45;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_45) : "h"((uint16_t)(_e4m3x2_to_f16x2_22 >> 16)));
                float _fdiv_rn_18 = __fdiv_rn(inv_1, _cvt_f32_f16_44);
                float _fdiv_rn_19 = __fdiv_rn(inv_1, _cvt_f32_f16_45);
                float m0_833 = 0.0f;
                float m1_834 = 0.0f;
                if (_cvt_f32_f16_44 > 0.0f) {
                    m0_833 = _fdiv_rn_18;
                }
                if (_cvt_f32_f16_45 > 0.0f) {
                    m1_834 = _fdiv_rn_19;
                }
                float2 mm_835 = make_float2(m0_833, m0_833);
                float2 pair_836 = make_float2(acc_407[0], acc_407[1]);
                float2 g_837;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_837) : "l"(*(const unsigned long long*)&pair_836), "l"(*(const unsigned long long*)&mm_835));
                uint32_t _fp4_pair_96;
                uint16_t _e2m1_pair_h_191;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_191) : "f"(g_837.x), "f"(g_837.y));
                _fp4_pair_96 = (uint32_t)_e2m1_pair_h_191;
                float2 pair_838 = make_float2(acc_407[2], acc_407[3]);
                float2 g_839;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_839) : "l"(*(const unsigned long long*)&pair_838), "l"(*(const unsigned long long*)&mm_835));
                uint32_t _fp4_pair_97;
                uint16_t _e2m1_pair_h_192;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_192) : "f"(g_839.x), "f"(g_839.y));
                _fp4_pair_97 = (uint32_t)_e2m1_pair_h_192;
                float2 pair_840 = make_float2(acc_407[4], acc_407[5]);
                float2 g_841;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_841) : "l"(*(const unsigned long long*)&pair_840), "l"(*(const unsigned long long*)&mm_835));
                uint32_t _fp4_pair_98;
                uint16_t _e2m1_pair_h_193;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_193) : "f"(g_841.x), "f"(g_841.y));
                _fp4_pair_98 = (uint32_t)_e2m1_pair_h_193;
                float2 pair_842 = make_float2(acc_407[6], acc_407[7]);
                float2 g_843;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_843) : "l"(*(const unsigned long long*)&pair_842), "l"(*(const unsigned long long*)&mm_835));
                uint32_t _fp4_e2m1_pack_pair_194;
                uint16_t _e2m1_pair_h_195;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_195) : "f"(g_843.x), "f"(g_843.y));
                _fp4_e2m1_pack_pair_194 = (uint32_t)_e2m1_pair_h_195;
                unsigned int word_844 = _fp4_pair_96 | _fp4_pair_97 << 8 | _fp4_pair_98 << 16 | _fp4_e2m1_pack_pair_194 << 24;
                unsigned int carry_845 = (word_844 & 2004318071) + 2004318071;
                q_830[0] = word_844 & (carry_845 | 2004318071);
                float2 mm_846 = make_float2(m0_833, m0_833);
                float2 pair_847 = make_float2(acc_407[8], acc_407[9]);
                float2 g_848;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_848) : "l"(*(const unsigned long long*)&pair_847), "l"(*(const unsigned long long*)&mm_846));
                uint32_t _fp4_pair_99;
                uint16_t _e2m1_pair_h_196;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_196) : "f"(g_848.x), "f"(g_848.y));
                _fp4_pair_99 = (uint32_t)_e2m1_pair_h_196;
                float2 pair_849 = make_float2(acc_407[10], acc_407[11]);
                float2 g_850;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_850) : "l"(*(const unsigned long long*)&pair_849), "l"(*(const unsigned long long*)&mm_846));
                uint32_t _fp4_pair_100;
                uint16_t _e2m1_pair_h_197;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_197) : "f"(g_850.x), "f"(g_850.y));
                _fp4_pair_100 = (uint32_t)_e2m1_pair_h_197;
                float2 pair_851 = make_float2(acc_407[12], acc_407[13]);
                float2 g_852;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_852) : "l"(*(const unsigned long long*)&pair_851), "l"(*(const unsigned long long*)&mm_846));
                uint32_t _fp4_pair_101;
                uint16_t _e2m1_pair_h_198;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_198) : "f"(g_852.x), "f"(g_852.y));
                _fp4_pair_101 = (uint32_t)_e2m1_pair_h_198;
                float2 pair_853 = make_float2(acc_407[14], acc_407[15]);
                float2 g_854;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_854) : "l"(*(const unsigned long long*)&pair_853), "l"(*(const unsigned long long*)&mm_846));
                uint32_t _fp4_e2m1_pack_pair_199;
                uint16_t _e2m1_pair_h_200;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_200) : "f"(g_854.x), "f"(g_854.y));
                _fp4_e2m1_pack_pair_199 = (uint32_t)_e2m1_pair_h_200;
                unsigned int word_855 = _fp4_pair_99 | _fp4_pair_100 << 8 | _fp4_pair_101 << 16 | _fp4_e2m1_pack_pair_199 << 24;
                unsigned int carry_856 = (word_855 & 2004318071) + 2004318071;
                q_830[1] = word_855 & (carry_856 | 2004318071);
                float2 mm_857 = make_float2(m1_834, m1_834);
                float2 pair_858 = make_float2(acc_407[16], acc_407[17]);
                float2 g_859;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_859) : "l"(*(const unsigned long long*)&pair_858), "l"(*(const unsigned long long*)&mm_857));
                uint32_t _fp4_pair_102;
                uint16_t _e2m1_pair_h_201;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_201) : "f"(g_859.x), "f"(g_859.y));
                _fp4_pair_102 = (uint32_t)_e2m1_pair_h_201;
                float2 pair_860 = make_float2(acc_407[18], acc_407[19]);
                float2 g_861;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_861) : "l"(*(const unsigned long long*)&pair_860), "l"(*(const unsigned long long*)&mm_857));
                uint32_t _fp4_pair_103;
                uint16_t _e2m1_pair_h_202;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_202) : "f"(g_861.x), "f"(g_861.y));
                _fp4_pair_103 = (uint32_t)_e2m1_pair_h_202;
                float2 pair_862 = make_float2(acc_407[20], acc_407[21]);
                float2 g_863;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_863) : "l"(*(const unsigned long long*)&pair_862), "l"(*(const unsigned long long*)&mm_857));
                uint32_t _fp4_pair_104;
                uint16_t _e2m1_pair_h_203;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_203) : "f"(g_863.x), "f"(g_863.y));
                _fp4_pair_104 = (uint32_t)_e2m1_pair_h_203;
                float2 pair_864 = make_float2(acc_407[22], acc_407[23]);
                float2 g_865;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_865) : "l"(*(const unsigned long long*)&pair_864), "l"(*(const unsigned long long*)&mm_857));
                uint32_t _fp4_e2m1_pack_pair_204;
                uint16_t _e2m1_pair_h_205;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_205) : "f"(g_865.x), "f"(g_865.y));
                _fp4_e2m1_pack_pair_204 = (uint32_t)_e2m1_pair_h_205;
                unsigned int word_866 = _fp4_pair_102 | _fp4_pair_103 << 8 | _fp4_pair_104 << 16 | _fp4_e2m1_pack_pair_204 << 24;
                unsigned int carry_867 = (word_866 & 2004318071) + 2004318071;
                q_830[2] = word_866 & (carry_867 | 2004318071);
                float2 mm_868 = make_float2(m1_834, m1_834);
                float2 pair_869 = make_float2(acc_407[24], acc_407[25]);
                float2 g_870;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_870) : "l"(*(const unsigned long long*)&pair_869), "l"(*(const unsigned long long*)&mm_868));
                uint32_t _fp4_pair_105;
                uint16_t _e2m1_pair_h_206;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_206) : "f"(g_870.x), "f"(g_870.y));
                _fp4_pair_105 = (uint32_t)_e2m1_pair_h_206;
                float2 pair_871 = make_float2(acc_407[26], acc_407[27]);
                float2 g_872;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_872) : "l"(*(const unsigned long long*)&pair_871), "l"(*(const unsigned long long*)&mm_868));
                uint32_t _fp4_pair_106;
                uint16_t _e2m1_pair_h_207;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_207) : "f"(g_872.x), "f"(g_872.y));
                _fp4_pair_106 = (uint32_t)_e2m1_pair_h_207;
                float2 pair_873 = make_float2(acc_407[28], acc_407[29]);
                float2 g_874;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_874) : "l"(*(const unsigned long long*)&pair_873), "l"(*(const unsigned long long*)&mm_868));
                uint32_t _fp4_pair_107;
                uint16_t _e2m1_pair_h_208;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_208) : "f"(g_874.x), "f"(g_874.y));
                _fp4_pair_107 = (uint32_t)_e2m1_pair_h_208;
                float2 pair_875 = make_float2(acc_407[30], acc_407[31]);
                float2 g_876;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_876) : "l"(*(const unsigned long long*)&pair_875), "l"(*(const unsigned long long*)&mm_868));
                uint32_t _fp4_e2m1_pack_pair_209;
                uint16_t _e2m1_pair_h_210;
                asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_210) : "f"(g_876.x), "f"(g_876.y));
                _fp4_e2m1_pack_pair_209 = (uint32_t)_e2m1_pair_h_210;
                unsigned int word_877 = _fp4_pair_105 | _fp4_pair_106 << 8 | _fp4_pair_107 << 16 | _fp4_e2m1_pack_pair_209 << 24;
                unsigned int carry_878 = (word_877 & 2004318071) + 2004318071;
                q_830[3] = word_877 & (carry_878 | 2004318071);
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(tob_off + 16 * pp2_829)), "r"(q_830[0]), "r"(q_830[1]), "r"(q_830[2]), "r"(q_830[3]) : "memory");
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 2 * pp2_829)), "h"((uint16_t)(_e4m3x2_f32_8)) : "memory");
                float d_879[32] = {0};
                uint32_t _e4m3x2_to_f16x2_23;
                asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_23) : "h"((uint16_t)(_e4m3x2_f32_8)));
                float _cvt_f32_f16_46;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_46) : "h"((uint16_t)(_e4m3x2_to_f16x2_23)));
                float _cvt_f32_f16_47;
                asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_47) : "h"((uint16_t)(_e4m3x2_to_f16x2_23 >> 16)));
                float _mul_28;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_cvt_f32_f16_46), "f"(rs2));
                float _mul_29;
                asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_cvt_f32_f16_47), "f"(rs2));
                float2 scale0_880 = make_float2(_mul_28, _mul_28);
                float2 scale1_881 = make_float2(_mul_29, _mul_29);
                float _unpack_fp4_56[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_830[0]));
                    (&_unpack_fp4_56[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_56[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_56[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_56[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_56[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_56[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_56[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_56[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_56[0]), "=f"(_unpack_fp4_56[1]), "=f"(_unpack_fp4_56[2]), "=f"(_unpack_fp4_56[3]), "=f"(_unpack_fp4_56[4]), "=f"(_unpack_fp4_56[5]), "=f"(_unpack_fp4_56[6]), "=f"(_unpack_fp4_56[7]) : "r"(q_830[0]));
                    #endif
                }
                float2 a_882 = make_float2(d_879[0], d_879[1]);
                float2 q_883 = make_float2(_unpack_fp4_56[0], _unpack_fp4_56[1]);
                float2 r_884 = fma_f32x2_rn_noftz(q_883, scale0_880, a_882);
                d_879[0] = r_884.x;
                d_879[1] = r_884.y;
                float2 a_885 = make_float2(d_879[2], d_879[3]);
                float2 q_886 = make_float2(_unpack_fp4_56[2], _unpack_fp4_56[3]);
                float2 r_887 = fma_f32x2_rn_noftz(q_886, scale0_880, a_885);
                d_879[2] = r_887.x;
                d_879[3] = r_887.y;
                float2 a_888 = make_float2(d_879[4], d_879[5]);
                float2 q_889 = make_float2(_unpack_fp4_56[4], _unpack_fp4_56[5]);
                float2 r_890 = fma_f32x2_rn_noftz(q_889, scale0_880, a_888);
                d_879[4] = r_890.x;
                d_879[5] = r_890.y;
                float2 a_891 = make_float2(d_879[6], d_879[7]);
                float2 q_892 = make_float2(_unpack_fp4_56[6], _unpack_fp4_56[7]);
                float2 r_893 = fma_f32x2_rn_noftz(q_892, scale0_880, a_891);
                d_879[6] = r_893.x;
                d_879[7] = r_893.y;
                float _unpack_fp4_57[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_830[1 + 0]));
                    (&_unpack_fp4_57[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_57[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_57[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_57[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_57[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_57[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_57[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_57[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_57[0]), "=f"(_unpack_fp4_57[1]), "=f"(_unpack_fp4_57[2]), "=f"(_unpack_fp4_57[3]), "=f"(_unpack_fp4_57[4]), "=f"(_unpack_fp4_57[5]), "=f"(_unpack_fp4_57[6]), "=f"(_unpack_fp4_57[7]) : "r"(q_830[1 + 0]));
                    #endif
                }
                float2 a_894 = make_float2(d_879[8], d_879[9]);
                float2 q_895 = make_float2(_unpack_fp4_57[0], _unpack_fp4_57[1]);
                float2 r_896 = fma_f32x2_rn_noftz(q_895, scale0_880, a_894);
                d_879[8] = r_896.x;
                d_879[9] = r_896.y;
                float2 a_897 = make_float2(d_879[10], d_879[11]);
                float2 q_898 = make_float2(_unpack_fp4_57[2], _unpack_fp4_57[3]);
                float2 r_899 = fma_f32x2_rn_noftz(q_898, scale0_880, a_897);
                d_879[10] = r_899.x;
                d_879[11] = r_899.y;
                float2 a_900 = make_float2(d_879[12], d_879[13]);
                float2 q_901 = make_float2(_unpack_fp4_57[4], _unpack_fp4_57[5]);
                float2 r_902 = fma_f32x2_rn_noftz(q_901, scale0_880, a_900);
                d_879[12] = r_902.x;
                d_879[13] = r_902.y;
                float2 a_903 = make_float2(d_879[14], d_879[15]);
                float2 q_904 = make_float2(_unpack_fp4_57[6], _unpack_fp4_57[7]);
                float2 r_905 = fma_f32x2_rn_noftz(q_904, scale0_880, a_903);
                d_879[14] = r_905.x;
                d_879[15] = r_905.y;
                float _unpack_fp4_58[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_830[2 + 0]));
                    (&_unpack_fp4_58[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_58[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_58[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_58[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_58[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_58[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_58[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_58[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_58[0]), "=f"(_unpack_fp4_58[1]), "=f"(_unpack_fp4_58[2]), "=f"(_unpack_fp4_58[3]), "=f"(_unpack_fp4_58[4]), "=f"(_unpack_fp4_58[5]), "=f"(_unpack_fp4_58[6]), "=f"(_unpack_fp4_58[7]) : "r"(q_830[2 + 0]));
                    #endif
                }
                float2 a_906 = make_float2(d_879[16], d_879[17]);
                float2 q_907 = make_float2(_unpack_fp4_58[0], _unpack_fp4_58[1]);
                float2 r_908 = fma_f32x2_rn_noftz(q_907, scale1_881, a_906);
                d_879[16] = r_908.x;
                d_879[17] = r_908.y;
                float2 a_909 = make_float2(d_879[18], d_879[19]);
                float2 q_910 = make_float2(_unpack_fp4_58[2], _unpack_fp4_58[3]);
                float2 r_911 = fma_f32x2_rn_noftz(q_910, scale1_881, a_909);
                d_879[18] = r_911.x;
                d_879[19] = r_911.y;
                float2 a_912 = make_float2(d_879[20], d_879[21]);
                float2 q_913 = make_float2(_unpack_fp4_58[4], _unpack_fp4_58[5]);
                float2 r_914 = fma_f32x2_rn_noftz(q_913, scale1_881, a_912);
                d_879[20] = r_914.x;
                d_879[21] = r_914.y;
                float2 a_915 = make_float2(d_879[22], d_879[23]);
                float2 q_916 = make_float2(_unpack_fp4_58[6], _unpack_fp4_58[7]);
                float2 r_917 = fma_f32x2_rn_noftz(q_916, scale1_881, a_915);
                d_879[22] = r_917.x;
                d_879[23] = r_917.y;
                float _unpack_fp4_59[8];
                {
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                    uint32_t _fp4_decode_pairs[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(q_830[3 + 0]));
                    (&_unpack_fp4_59[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                    (&_unpack_fp4_59[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                    (&_unpack_fp4_59[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                    (&_unpack_fp4_59[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                    (&_unpack_fp4_59[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                    (&_unpack_fp4_59[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                    (&_unpack_fp4_59[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                    (&_unpack_fp4_59[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                    #else
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_59[0]), "=f"(_unpack_fp4_59[1]), "=f"(_unpack_fp4_59[2]), "=f"(_unpack_fp4_59[3]), "=f"(_unpack_fp4_59[4]), "=f"(_unpack_fp4_59[5]), "=f"(_unpack_fp4_59[6]), "=f"(_unpack_fp4_59[7]) : "r"(q_830[3 + 0]));
                    #endif
                }
                float2 a_918 = make_float2(d_879[24], d_879[25]);
                float2 q_919 = make_float2(_unpack_fp4_59[0], _unpack_fp4_59[1]);
                float2 r_920 = fma_f32x2_rn_noftz(q_919, scale1_881, a_918);
                d_879[24] = r_920.x;
                d_879[25] = r_920.y;
                float2 a_921 = make_float2(d_879[26], d_879[27]);
                float2 q_922 = make_float2(_unpack_fp4_59[2], _unpack_fp4_59[3]);
                float2 r_923 = fma_f32x2_rn_noftz(q_922, scale1_881, a_921);
                d_879[26] = r_923.x;
                d_879[27] = r_923.y;
                float2 a_924 = make_float2(d_879[28], d_879[29]);
                float2 q_925 = make_float2(_unpack_fp4_59[4], _unpack_fp4_59[5]);
                float2 r_926 = fma_f32x2_rn_noftz(q_925, scale1_881, a_924);
                d_879[28] = r_926.x;
                d_879[29] = r_926.y;
                float2 a_927 = make_float2(d_879[30], d_879[31]);
                float2 q_928 = make_float2(_unpack_fp4_59[6], _unpack_fp4_59[7]);
                float2 r_929 = fma_f32x2_rn_noftz(q_928, scale1_881, a_927);
                d_879[30] = r_929.x;
                d_879[31] = r_929.y;
                unsigned int d_879_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(d_879[_lp*2 + 0], d_879[_lp*2+1 + 0]));
                    d_879_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(d_879_bf16[0]), "r"(d_879_bf16[1]), "r"(d_879_bf16[2]), "r"(d_879_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(d_879_bf16[4]), "r"(d_879_bf16[5]), "r"(d_879_bf16[6]), "r"(d_879_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(d_879_bf16[8]), "r"(d_879_bf16[9]), "r"(d_879_bf16[10]), "r"(d_879_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(d_879_bf16[12]), "r"(d_879_bf16[13]), "r"(d_879_bf16[14]), "r"(d_879_bf16[15]) : "memory");
                __syncwarp();
                int c_930 = lane_1;
                int l_931 = c_930 >> 2;
                unsigned int w4_932[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_932[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_932[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_932[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_932[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_931 + (c_930 & 3 ^ l_931 >> 1 & 3)))));
                long long idx_933 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_930));
                {
                    int4 _iv4 = make_int4(w4_932[0 + 0], w4_932[0 + 1], w4_932[0 + 2], w4_932[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_933) = _iv4;
                }
                int c_934 = lane_1 + 32;
                int l_935 = c_934 >> 2;
                unsigned int w4_936[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_936[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_936[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_936[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_936[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_935 + (c_934 & 3 ^ l_935 >> 1 & 3)))));
                long long idx_937 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_934));
                {
                    int4 _iv4 = make_int4(w4_936[0 + 0], w4_936[0 + 1], w4_936[0 + 2], w4_936[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_937) = _iv4;
                }
                int c_938 = lane_1 + 64;
                int l_939 = c_938 >> 2;
                unsigned int w4_940[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_940[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_940[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_940[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_940[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_939 + (c_938 & 3 ^ l_939 >> 1 & 3)))));
                long long idx_941 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_938));
                {
                    int4 _iv4 = make_int4(w4_940[0 + 0], w4_940[0 + 1], w4_940[0 + 2], w4_940[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_941) = _iv4;
                }
                int c_942 = lane_1 + 96;
                int l_943 = c_942 >> 2;
                unsigned int w4_944[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&w4_944[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_944[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_944[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_944[(0) + 3]))
                    : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_943 + (c_942 & 3 ^ l_943 >> 1 & 3)))));
                long long idx_945 = trow64 * 3072 + (long long)(4 * (128 * (3 * h + 2) + c_942));
                {
                    int4 _iv4 = make_int4(w4_944[0 + 0], w4_944[0 + 1], w4_944[0 + 2], w4_944[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_945) = _iv4;
                }
                __syncwarp();
                if (h == 0 && lane_1 < 4) {
                    {
                        uint32_t _addr_211 = static_cast<uint32_t>(wsm_addr + (unsigned int)(tob_off + 3072 + 384 + 4 * lane_1));
                        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_211), "f"(rs2) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("bar.sync %0, 64;" :: "r"(1 + t) : "memory");
                if (h == 0 && lane_1 == 0) {
                    int ag_wo = (int)ag_w;
                    int ag_ro = (int)ag_r;
                    int capo_0 = (int)capo64;
                    int ix_1_1 = (int)tix64;
                    int rk_2 = (int)rank64;
                    int v_off_1 = ag_wo + (rk_2 * capo_0 + ix_1_1) * 3072;
                    int sc_off_1 = ag_wo + 4 * capo_0 * 3072 + (rk_2 * capo_0 + ix_1_1) * 384;
                    int m_off_1 = ag_ro + (rk_2 * capo_0 + ix_1_1) * 16;
                    int peer = (rank + 1) % 4;
                    uint8_t* pw = reinterpret_cast<uint8_t*>(tab[peer]);
                    uint8_t* pr = reinterpret_cast<uint8_t*>(tab[4 + peer]);
                    {
                        void* _cpbulk_dst_212 = reinterpret_cast<void*>(pw + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_212), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_213 = reinterpret_cast<void*>(pw + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_213), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_214 = reinterpret_cast<void*>(pr + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_214), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_3 = (rank + 2) % 4;
                    uint8_t* pw_4 = reinterpret_cast<uint8_t*>(tab[peer_3]);
                    uint8_t* pr_5 = reinterpret_cast<uint8_t*>(tab[4 + peer_3]);
                    {
                        void* _cpbulk_dst_215 = reinterpret_cast<void*>(pw_4 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_215), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_216 = reinterpret_cast<void*>(pw_4 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_216), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_217 = reinterpret_cast<void*>(pr_5 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_217), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
                            : "memory");
                    }
                    int peer_6 = (rank + 3) % 4;
                    uint8_t* pw_7 = reinterpret_cast<uint8_t*>(tab[peer_6]);
                    uint8_t* pr_8 = reinterpret_cast<uint8_t*>(tab[4 + peer_6]);
                    {
                        void* _cpbulk_dst_218 = reinterpret_cast<void*>(pw_7 + v_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_218), "r"(wsm_addr + (unsigned int)tob_off), "r"((uint32_t)(3072))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_219 = reinterpret_cast<void*>(pw_7 + sc_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_219), "r"(wsm_addr + (unsigned int)(tob_off + 3072)), "r"((uint32_t)(384))
                            : "memory");
                    }
                    {
                        void* _cpbulk_dst_220 = reinterpret_cast<void*>(pr_8 + m_off_1);
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_220), "r"(wsm_addr + (unsigned int)(tob_off + 3072 + 384)), "r"((uint32_t)(16))
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
                #pragma unroll 1
                for (int jj = 0; jj < 2; jj++) {
                    int pp0 = lane_1 + 96 * jj;
                    int pp1 = pp0 + 32;
                    int pp2_2 = pp0 + 64;
                    uint32_t _fp4_wire_values_1[12];
                    uint32_t _fp4_wire_scales_1[3];
                    uint16_t _wire_h16_221_0;
                    uint16_t _wire_h16_221_1;
                    uint16_t _wire_h16_221_2;
                    uint32_t _wire_vpend_221 = 7u;
                    uint32_t _wire_spend_221 = 7u;
                    do {
                        if (_wire_vpend_221 & 1u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp0)))) : "memory");
                        }
                        if (_wire_vpend_221 & 2u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp1)))) : "memory");
                        }
                        if (_wire_vpend_221 & 4u) {
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[8]), "=r"(_fp4_wire_values_1[9]), "=r"(_fp4_wire_values_1[10]), "=r"(_fp4_wire_values_1[11]) : "l"(reinterpret_cast<unsigned int*>(ts_wire + (v_1 + (int)(16 * pp2_2)))) : "memory");
                        }
                        if (_wire_spend_221 & 1u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_0) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp0)))) : "memory");
                            _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_221_0;
                        }
                        if (_wire_spend_221 & 2u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_1) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp1)))) : "memory");
                            _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_221_1;
                        }
                        if (_wire_spend_221 & 4u) {
                            asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_221_2) : "l"(reinterpret_cast<uint16_t*>(ts_wire + (sc_1 + (int)(2 * pp2_2)))) : "memory");
                            _fp4_wire_scales_1[2] = (uint32_t)_wire_h16_221_2;
                        }
                        _wire_vpend_221 = ((uint32_t)((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1) | ((uint32_t)((((_fp4_wire_values_1[8] & ~((_fp4_wire_values_1[8] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[9] & ~((_fp4_wire_values_1[9] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[10] & ~((_fp4_wire_values_1[10] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[11] & ~((_fp4_wire_values_1[11] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 2);
                        _wire_spend_221 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[2] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[2] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 2);
                    } while ((_wire_vpend_221 | _wire_spend_221) != 0u);
                    float acc_1[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_24;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_24) : "h"((uint16_t)(_fp4_wire_scales_1[0])));
                    float _cvt_f32_f16_48;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_48) : "h"((uint16_t)(_e4m3x2_to_f16x2_24)));
                    float _cvt_f32_f16_49;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_49) : "h"((uint16_t)(_e4m3x2_to_f16x2_24 >> 16)));
                    float _mul_30;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_cvt_f32_f16_48), "f"(rs_2));
                    float _mul_31;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_cvt_f32_f16_49), "f"(rs_2));
                    float2 scale0_1 = make_float2(_mul_30, _mul_30);
                    float2 scale1_1 = make_float2(_mul_31, _mul_31);
                    float _unpack_fp4_60[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[0]));
                        (&_unpack_fp4_60[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_60[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_60[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_60[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_60[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_60[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_60[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_60[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_60[0]), "=f"(_unpack_fp4_60[1]), "=f"(_unpack_fp4_60[2]), "=f"(_unpack_fp4_60[3]), "=f"(_unpack_fp4_60[4]), "=f"(_unpack_fp4_60[5]), "=f"(_unpack_fp4_60[6]), "=f"(_unpack_fp4_60[7]) : "r"(_fp4_wire_values_1[0]));
                        #endif
                    }
                    float2 a_3 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_1 = make_float2(_unpack_fp4_60[0], _unpack_fp4_60[1]);
                    float2 r_1 = fma_f32x2_rn_noftz(q_1, scale0_1, a_3);
                    acc_1[0] = r_1.x;
                    acc_1[1] = r_1.y;
                    float2 a_0 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_1_1 = make_float2(_unpack_fp4_60[2], _unpack_fp4_60[3]);
                    float2 r_2 = fma_f32x2_rn_noftz(q_1_1, scale0_1, a_0);
                    acc_1[2] = r_2.x;
                    acc_1[3] = r_2.y;
                    float2 a_3_1 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_4 = make_float2(_unpack_fp4_60[4], _unpack_fp4_60[5]);
                    float2 r_5 = fma_f32x2_rn_noftz(q_4, scale0_1, a_3_1);
                    acc_1[4] = r_5.x;
                    acc_1[5] = r_5.y;
                    float2 a_6 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_7 = make_float2(_unpack_fp4_60[6], _unpack_fp4_60[7]);
                    float2 r_8 = fma_f32x2_rn_noftz(q_7, scale0_1, a_6);
                    acc_1[6] = r_8.x;
                    acc_1[7] = r_8.y;
                    float _unpack_fp4_61[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[1 + 0]));
                        (&_unpack_fp4_61[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_61[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_61[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_61[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_61[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_61[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_61[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_61[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_61[0]), "=f"(_unpack_fp4_61[1]), "=f"(_unpack_fp4_61[2]), "=f"(_unpack_fp4_61[3]), "=f"(_unpack_fp4_61[4]), "=f"(_unpack_fp4_61[5]), "=f"(_unpack_fp4_61[6]), "=f"(_unpack_fp4_61[7]) : "r"(_fp4_wire_values_1[1 + 0]));
                        #endif
                    }
                    float2 a_9 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_10 = make_float2(_unpack_fp4_61[0], _unpack_fp4_61[1]);
                    float2 r_11 = fma_f32x2_rn_noftz(q_10, scale0_1, a_9);
                    acc_1[8] = r_11.x;
                    acc_1[9] = r_11.y;
                    float2 a_12_1 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_13 = make_float2(_unpack_fp4_61[2], _unpack_fp4_61[3]);
                    float2 r_14 = fma_f32x2_rn_noftz(q_13, scale0_1, a_12_1);
                    acc_1[10] = r_14.x;
                    acc_1[11] = r_14.y;
                    float2 a_15 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_16 = make_float2(_unpack_fp4_61[4], _unpack_fp4_61[5]);
                    float2 r_17 = fma_f32x2_rn_noftz(q_16, scale0_1, a_15);
                    acc_1[12] = r_17.x;
                    acc_1[13] = r_17.y;
                    float2 a_18 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_19 = make_float2(_unpack_fp4_61[6], _unpack_fp4_61[7]);
                    float2 r_20 = fma_f32x2_rn_noftz(q_19, scale0_1, a_18);
                    acc_1[14] = r_20.x;
                    acc_1[15] = r_20.y;
                    float _unpack_fp4_62[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[2 + 0]));
                        (&_unpack_fp4_62[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_62[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_62[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_62[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_62[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_62[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_62[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_62[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_62[0]), "=f"(_unpack_fp4_62[1]), "=f"(_unpack_fp4_62[2]), "=f"(_unpack_fp4_62[3]), "=f"(_unpack_fp4_62[4]), "=f"(_unpack_fp4_62[5]), "=f"(_unpack_fp4_62[6]), "=f"(_unpack_fp4_62[7]) : "r"(_fp4_wire_values_1[2 + 0]));
                        #endif
                    }
                    float2 a_21 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_22 = make_float2(_unpack_fp4_62[0], _unpack_fp4_62[1]);
                    float2 r_23 = fma_f32x2_rn_noftz(q_22, scale1_1, a_21);
                    acc_1[16] = r_23.x;
                    acc_1[17] = r_23.y;
                    float2 a_24 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_25 = make_float2(_unpack_fp4_62[2], _unpack_fp4_62[3]);
                    float2 r_26 = fma_f32x2_rn_noftz(q_25, scale1_1, a_24);
                    acc_1[18] = r_26.x;
                    acc_1[19] = r_26.y;
                    float2 a_27 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_28 = make_float2(_unpack_fp4_62[4], _unpack_fp4_62[5]);
                    float2 r_29 = fma_f32x2_rn_noftz(q_28, scale1_1, a_27);
                    acc_1[20] = r_29.x;
                    acc_1[21] = r_29.y;
                    float2 a_30 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_31 = make_float2(_unpack_fp4_62[6], _unpack_fp4_62[7]);
                    float2 r_32 = fma_f32x2_rn_noftz(q_31, scale1_1, a_30);
                    acc_1[22] = r_32.x;
                    acc_1[23] = r_32.y;
                    float _unpack_fp4_63[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[3 + 0]));
                        (&_unpack_fp4_63[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_63[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_63[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_63[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_63[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_63[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_63[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_63[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_63[0]), "=f"(_unpack_fp4_63[1]), "=f"(_unpack_fp4_63[2]), "=f"(_unpack_fp4_63[3]), "=f"(_unpack_fp4_63[4]), "=f"(_unpack_fp4_63[5]), "=f"(_unpack_fp4_63[6]), "=f"(_unpack_fp4_63[7]) : "r"(_fp4_wire_values_1[3 + 0]));
                        #endif
                    }
                    float2 a_33 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_34 = make_float2(_unpack_fp4_63[0], _unpack_fp4_63[1]);
                    float2 r_35 = fma_f32x2_rn_noftz(q_34, scale1_1, a_33);
                    acc_1[24] = r_35.x;
                    acc_1[25] = r_35.y;
                    float2 a_36 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_37 = make_float2(_unpack_fp4_63[2], _unpack_fp4_63[3]);
                    float2 r_38 = fma_f32x2_rn_noftz(q_37, scale1_1, a_36);
                    acc_1[26] = r_38.x;
                    acc_1[27] = r_38.y;
                    float2 a_39 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_40 = make_float2(_unpack_fp4_63[4], _unpack_fp4_63[5]);
                    float2 r_41 = fma_f32x2_rn_noftz(q_40, scale1_1, a_39);
                    acc_1[28] = r_41.x;
                    acc_1[29] = r_41.y;
                    float2 a_42_1 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_43 = make_float2(_unpack_fp4_63[6], _unpack_fp4_63[7]);
                    float2 r_44 = fma_f32x2_rn_noftz(q_43, scale1_1, a_42_1);
                    acc_1[30] = r_44.x;
                    acc_1[31] = r_44.y;
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
                    int c_45 = lane_1;
                    int l_1 = c_45 >> 2;
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_1 + (c_45 & 3 ^ l_1 >> 1 & 3)))));
                    long long idx_1 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_45));
                    {
                        int4 _iv4 = make_int4(w4_1[0 + 0], w4_1[0 + 1], w4_1[0 + 2], w4_1[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_1) = _iv4;
                    }
                    int c_46 = lane_1 + 32;
                    int l_47 = c_46 >> 2;
                    unsigned int w4_48[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_48[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_47 + (c_46 & 3 ^ l_47 >> 1 & 3)))));
                    long long idx_49 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_46));
                    {
                        int4 _iv4 = make_int4(w4_48[0 + 0], w4_48[0 + 1], w4_48[0 + 2], w4_48[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_49) = _iv4;
                    }
                    int c_50 = lane_1 + 64;
                    int l_51 = c_50 >> 2;
                    unsigned int w4_52[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_52[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_51 + (c_50 & 3 ^ l_51 >> 1 & 3)))));
                    long long idx_53 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_50));
                    {
                        int4 _iv4 = make_int4(w4_52[0 + 0], w4_52[0 + 1], w4_52[0 + 2], w4_52[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_53) = _iv4;
                    }
                    int c_54 = lane_1 + 96;
                    int l_55 = c_54 >> 2;
                    unsigned int w4_56[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_56[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_55 + (c_54 & 3 ^ l_55 >> 1 & 3)))));
                    long long idx_57 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj) + c_54));
                    {
                        int4 _iv4 = make_int4(w4_56[0 + 0], w4_56[0 + 1], w4_56[0 + 2], w4_56[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_57) = _iv4;
                    }
                    __syncwarp();
                    float acc_58[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_25;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_25) : "h"((uint16_t)(_fp4_wire_scales_1[1])));
                    float _cvt_f32_f16_50;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_50) : "h"((uint16_t)(_e4m3x2_to_f16x2_25)));
                    float _cvt_f32_f16_51;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_51) : "h"((uint16_t)(_e4m3x2_to_f16x2_25 >> 16)));
                    float _mul_32;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_cvt_f32_f16_50), "f"(rs_2));
                    float _mul_33;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_cvt_f32_f16_51), "f"(rs_2));
                    float2 scale0_59 = make_float2(_mul_32, _mul_32);
                    float2 scale1_60 = make_float2(_mul_33, _mul_33);
                    float _unpack_fp4_64[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[4 + 0]));
                        (&_unpack_fp4_64[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_64[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_64[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_64[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_64[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_64[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_64[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_64[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_64[0]), "=f"(_unpack_fp4_64[1]), "=f"(_unpack_fp4_64[2]), "=f"(_unpack_fp4_64[3]), "=f"(_unpack_fp4_64[4]), "=f"(_unpack_fp4_64[5]), "=f"(_unpack_fp4_64[6]), "=f"(_unpack_fp4_64[7]) : "r"(_fp4_wire_values_1[4 + 0]));
                        #endif
                    }
                    float2 a_61_1 = make_float2(acc_58[0], acc_58[1]);
                    float2 q_62_1 = make_float2(_unpack_fp4_64[0], _unpack_fp4_64[1]);
                    float2 r_63_1 = fma_f32x2_rn_noftz(q_62_1, scale0_59, a_61_1);
                    acc_58[0] = r_63_1.x;
                    acc_58[1] = r_63_1.y;
                    float2 a_64_1 = make_float2(acc_58[2], acc_58[3]);
                    float2 q_65_1 = make_float2(_unpack_fp4_64[2], _unpack_fp4_64[3]);
                    float2 r_66_1 = fma_f32x2_rn_noftz(q_65_1, scale0_59, a_64_1);
                    acc_58[2] = r_66_1.x;
                    acc_58[3] = r_66_1.y;
                    float2 a_67_1 = make_float2(acc_58[4], acc_58[5]);
                    float2 q_68_1 = make_float2(_unpack_fp4_64[4], _unpack_fp4_64[5]);
                    float2 r_69_1 = fma_f32x2_rn_noftz(q_68_1, scale0_59, a_67_1);
                    acc_58[4] = r_69_1.x;
                    acc_58[5] = r_69_1.y;
                    float2 a_70_1 = make_float2(acc_58[6], acc_58[7]);
                    float2 q_71_1 = make_float2(_unpack_fp4_64[6], _unpack_fp4_64[7]);
                    float2 r_72_1 = fma_f32x2_rn_noftz(q_71_1, scale0_59, a_70_1);
                    acc_58[6] = r_72_1.x;
                    acc_58[7] = r_72_1.y;
                    float _unpack_fp4_65[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[5 + 0]));
                        (&_unpack_fp4_65[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_65[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_65[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_65[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_65[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_65[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_65[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_65[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_65[0]), "=f"(_unpack_fp4_65[1]), "=f"(_unpack_fp4_65[2]), "=f"(_unpack_fp4_65[3]), "=f"(_unpack_fp4_65[4]), "=f"(_unpack_fp4_65[5]), "=f"(_unpack_fp4_65[6]), "=f"(_unpack_fp4_65[7]) : "r"(_fp4_wire_values_1[5 + 0]));
                        #endif
                    }
                    float2 a_73_1 = make_float2(acc_58[8], acc_58[9]);
                    float2 q_74_1 = make_float2(_unpack_fp4_65[0], _unpack_fp4_65[1]);
                    float2 r_75_1 = fma_f32x2_rn_noftz(q_74_1, scale0_59, a_73_1);
                    acc_58[8] = r_75_1.x;
                    acc_58[9] = r_75_1.y;
                    float2 a_76_1 = make_float2(acc_58[10], acc_58[11]);
                    float2 q_77_1 = make_float2(_unpack_fp4_65[2], _unpack_fp4_65[3]);
                    float2 r_78_1 = fma_f32x2_rn_noftz(q_77_1, scale0_59, a_76_1);
                    acc_58[10] = r_78_1.x;
                    acc_58[11] = r_78_1.y;
                    float2 a_79_1 = make_float2(acc_58[12], acc_58[13]);
                    float2 q_80_1 = make_float2(_unpack_fp4_65[4], _unpack_fp4_65[5]);
                    float2 r_81_1 = fma_f32x2_rn_noftz(q_80_1, scale0_59, a_79_1);
                    acc_58[12] = r_81_1.x;
                    acc_58[13] = r_81_1.y;
                    float2 a_82_1 = make_float2(acc_58[14], acc_58[15]);
                    float2 q_83_1 = make_float2(_unpack_fp4_65[6], _unpack_fp4_65[7]);
                    float2 r_84_1 = fma_f32x2_rn_noftz(q_83_1, scale0_59, a_82_1);
                    acc_58[14] = r_84_1.x;
                    acc_58[15] = r_84_1.y;
                    float _unpack_fp4_66[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[6 + 0]));
                        (&_unpack_fp4_66[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_66[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_66[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_66[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_66[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_66[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_66[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_66[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_66[0]), "=f"(_unpack_fp4_66[1]), "=f"(_unpack_fp4_66[2]), "=f"(_unpack_fp4_66[3]), "=f"(_unpack_fp4_66[4]), "=f"(_unpack_fp4_66[5]), "=f"(_unpack_fp4_66[6]), "=f"(_unpack_fp4_66[7]) : "r"(_fp4_wire_values_1[6 + 0]));
                        #endif
                    }
                    float2 a_85_1 = make_float2(acc_58[16], acc_58[17]);
                    float2 q_86_1 = make_float2(_unpack_fp4_66[0], _unpack_fp4_66[1]);
                    float2 r_87_1 = fma_f32x2_rn_noftz(q_86_1, scale1_60, a_85_1);
                    acc_58[16] = r_87_1.x;
                    acc_58[17] = r_87_1.y;
                    float2 a_88_1 = make_float2(acc_58[18], acc_58[19]);
                    float2 q_89_1 = make_float2(_unpack_fp4_66[2], _unpack_fp4_66[3]);
                    float2 r_90_1 = fma_f32x2_rn_noftz(q_89_1, scale1_60, a_88_1);
                    acc_58[18] = r_90_1.x;
                    acc_58[19] = r_90_1.y;
                    float2 a_91_1 = make_float2(acc_58[20], acc_58[21]);
                    float2 q_92_1 = make_float2(_unpack_fp4_66[4], _unpack_fp4_66[5]);
                    float2 r_93_1 = fma_f32x2_rn_noftz(q_92_1, scale1_60, a_91_1);
                    acc_58[20] = r_93_1.x;
                    acc_58[21] = r_93_1.y;
                    float2 a_94_1 = make_float2(acc_58[22], acc_58[23]);
                    float2 q_95_1 = make_float2(_unpack_fp4_66[6], _unpack_fp4_66[7]);
                    float2 r_96_1 = fma_f32x2_rn_noftz(q_95_1, scale1_60, a_94_1);
                    acc_58[22] = r_96_1.x;
                    acc_58[23] = r_96_1.y;
                    float _unpack_fp4_67[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[7 + 0]));
                        (&_unpack_fp4_67[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_67[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_67[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_67[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_67[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_67[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_67[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_67[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_67[0]), "=f"(_unpack_fp4_67[1]), "=f"(_unpack_fp4_67[2]), "=f"(_unpack_fp4_67[3]), "=f"(_unpack_fp4_67[4]), "=f"(_unpack_fp4_67[5]), "=f"(_unpack_fp4_67[6]), "=f"(_unpack_fp4_67[7]) : "r"(_fp4_wire_values_1[7 + 0]));
                        #endif
                    }
                    float2 a_97_1 = make_float2(acc_58[24], acc_58[25]);
                    float2 q_98_1 = make_float2(_unpack_fp4_67[0], _unpack_fp4_67[1]);
                    float2 r_99_1 = fma_f32x2_rn_noftz(q_98_1, scale1_60, a_97_1);
                    acc_58[24] = r_99_1.x;
                    acc_58[25] = r_99_1.y;
                    float2 a_100_1 = make_float2(acc_58[26], acc_58[27]);
                    float2 q_101_1 = make_float2(_unpack_fp4_67[2], _unpack_fp4_67[3]);
                    float2 r_102_1 = fma_f32x2_rn_noftz(q_101_1, scale1_60, a_100_1);
                    acc_58[26] = r_102_1.x;
                    acc_58[27] = r_102_1.y;
                    float2 a_103_1 = make_float2(acc_58[28], acc_58[29]);
                    float2 q_104_1 = make_float2(_unpack_fp4_67[4], _unpack_fp4_67[5]);
                    float2 r_105_1 = fma_f32x2_rn_noftz(q_104_1, scale1_60, a_103_1);
                    acc_58[28] = r_105_1.x;
                    acc_58[29] = r_105_1.y;
                    float2 a_106 = make_float2(acc_58[30], acc_58[31]);
                    float2 q_107 = make_float2(_unpack_fp4_67[6], _unpack_fp4_67[7]);
                    float2 r_108 = fma_f32x2_rn_noftz(q_107, scale1_60, a_106);
                    acc_58[30] = r_108.x;
                    acc_58[31] = r_108.y;
                    unsigned int acc_58_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_58[_lp*2 + 0], acc_58[_lp*2+1 + 0]));
                        acc_58_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_58_bf16[0]), "r"(acc_58_bf16[1]), "r"(acc_58_bf16[2]), "r"(acc_58_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_58_bf16[4]), "r"(acc_58_bf16[5]), "r"(acc_58_bf16[6]), "r"(acc_58_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_58_bf16[8]), "r"(acc_58_bf16[9]), "r"(acc_58_bf16[10]), "r"(acc_58_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_58_bf16[12]), "r"(acc_58_bf16[13]), "r"(acc_58_bf16[14]), "r"(acc_58_bf16[15]) : "memory");
                    __syncwarp();
                    int c_109 = lane_1;
                    int l_110 = c_109 >> 2;
                    unsigned int w4_111[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_111[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_111[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_110 + (c_109 & 3 ^ l_110 >> 1 & 3)))));
                    long long idx_112 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_109));
                    {
                        int4 _iv4 = make_int4(w4_111[0 + 0], w4_111[0 + 1], w4_111[0 + 2], w4_111[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_112) = _iv4;
                    }
                    int c_113 = lane_1 + 32;
                    int l_114 = c_113 >> 2;
                    unsigned int w4_115[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_115[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_115[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_114 + (c_113 & 3 ^ l_114 >> 1 & 3)))));
                    long long idx_116 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_113));
                    {
                        int4 _iv4 = make_int4(w4_115[0 + 0], w4_115[0 + 1], w4_115[0 + 2], w4_115[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_116) = _iv4;
                    }
                    int c_117 = lane_1 + 64;
                    int l_118 = c_117 >> 2;
                    unsigned int w4_119[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_119[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_119[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_118 + (c_117 & 3 ^ l_118 >> 1 & 3)))));
                    long long idx_120 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_117));
                    {
                        int4 _iv4 = make_int4(w4_119[0 + 0], w4_119[0 + 1], w4_119[0 + 2], w4_119[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_120) = _iv4;
                    }
                    int c_121 = lane_1 + 96;
                    int l_122 = c_121 >> 2;
                    unsigned int w4_123[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_123[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_123[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_122 + (c_121 & 3 ^ l_122 >> 1 & 3)))));
                    long long idx_124 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 1) + c_121));
                    {
                        int4 _iv4 = make_int4(w4_123[0 + 0], w4_123[0 + 1], w4_123[0 + 2], w4_123[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_124) = _iv4;
                    }
                    __syncwarp();
                    float acc_125[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_26;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_26) : "h"((uint16_t)(_fp4_wire_scales_1[2])));
                    float _cvt_f32_f16_52;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_52) : "h"((uint16_t)(_e4m3x2_to_f16x2_26)));
                    float _cvt_f32_f16_53;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_53) : "h"((uint16_t)(_e4m3x2_to_f16x2_26 >> 16)));
                    float _mul_34;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_cvt_f32_f16_52), "f"(rs_2));
                    float _mul_35;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_cvt_f32_f16_53), "f"(rs_2));
                    float2 scale0_126 = make_float2(_mul_34, _mul_34);
                    float2 scale1_127 = make_float2(_mul_35, _mul_35);
                    float _unpack_fp4_68[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[8 + 0]));
                        (&_unpack_fp4_68[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_68[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_68[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_68[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_68[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_68[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_68[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_68[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_68[0]), "=f"(_unpack_fp4_68[1]), "=f"(_unpack_fp4_68[2]), "=f"(_unpack_fp4_68[3]), "=f"(_unpack_fp4_68[4]), "=f"(_unpack_fp4_68[5]), "=f"(_unpack_fp4_68[6]), "=f"(_unpack_fp4_68[7]) : "r"(_fp4_wire_values_1[8 + 0]));
                        #endif
                    }
                    float2 a_128 = make_float2(acc_125[0], acc_125[1]);
                    float2 q_129 = make_float2(_unpack_fp4_68[0], _unpack_fp4_68[1]);
                    float2 r_130 = fma_f32x2_rn_noftz(q_129, scale0_126, a_128);
                    acc_125[0] = r_130.x;
                    acc_125[1] = r_130.y;
                    float2 a_131 = make_float2(acc_125[2], acc_125[3]);
                    float2 q_132 = make_float2(_unpack_fp4_68[2], _unpack_fp4_68[3]);
                    float2 r_133 = fma_f32x2_rn_noftz(q_132, scale0_126, a_131);
                    acc_125[2] = r_133.x;
                    acc_125[3] = r_133.y;
                    float2 a_134 = make_float2(acc_125[4], acc_125[5]);
                    float2 q_135 = make_float2(_unpack_fp4_68[4], _unpack_fp4_68[5]);
                    float2 r_136 = fma_f32x2_rn_noftz(q_135, scale0_126, a_134);
                    acc_125[4] = r_136.x;
                    acc_125[5] = r_136.y;
                    float2 a_137 = make_float2(acc_125[6], acc_125[7]);
                    float2 q_138 = make_float2(_unpack_fp4_68[6], _unpack_fp4_68[7]);
                    float2 r_139 = fma_f32x2_rn_noftz(q_138, scale0_126, a_137);
                    acc_125[6] = r_139.x;
                    acc_125[7] = r_139.y;
                    float _unpack_fp4_69[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[9 + 0]));
                        (&_unpack_fp4_69[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_69[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_69[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_69[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_69[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_69[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_69[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_69[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_69[0]), "=f"(_unpack_fp4_69[1]), "=f"(_unpack_fp4_69[2]), "=f"(_unpack_fp4_69[3]), "=f"(_unpack_fp4_69[4]), "=f"(_unpack_fp4_69[5]), "=f"(_unpack_fp4_69[6]), "=f"(_unpack_fp4_69[7]) : "r"(_fp4_wire_values_1[9 + 0]));
                        #endif
                    }
                    float2 a_140 = make_float2(acc_125[8], acc_125[9]);
                    float2 q_141 = make_float2(_unpack_fp4_69[0], _unpack_fp4_69[1]);
                    float2 r_142 = fma_f32x2_rn_noftz(q_141, scale0_126, a_140);
                    acc_125[8] = r_142.x;
                    acc_125[9] = r_142.y;
                    float2 a_143 = make_float2(acc_125[10], acc_125[11]);
                    float2 q_144 = make_float2(_unpack_fp4_69[2], _unpack_fp4_69[3]);
                    float2 r_145 = fma_f32x2_rn_noftz(q_144, scale0_126, a_143);
                    acc_125[10] = r_145.x;
                    acc_125[11] = r_145.y;
                    float2 a_146 = make_float2(acc_125[12], acc_125[13]);
                    float2 q_147 = make_float2(_unpack_fp4_69[4], _unpack_fp4_69[5]);
                    float2 r_148 = fma_f32x2_rn_noftz(q_147, scale0_126, a_146);
                    acc_125[12] = r_148.x;
                    acc_125[13] = r_148.y;
                    float2 a_149 = make_float2(acc_125[14], acc_125[15]);
                    float2 q_150 = make_float2(_unpack_fp4_69[6], _unpack_fp4_69[7]);
                    float2 r_151 = fma_f32x2_rn_noftz(q_150, scale0_126, a_149);
                    acc_125[14] = r_151.x;
                    acc_125[15] = r_151.y;
                    float _unpack_fp4_70[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[10 + 0]));
                        (&_unpack_fp4_70[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_70[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_70[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_70[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_70[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_70[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_70[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_70[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_70[0]), "=f"(_unpack_fp4_70[1]), "=f"(_unpack_fp4_70[2]), "=f"(_unpack_fp4_70[3]), "=f"(_unpack_fp4_70[4]), "=f"(_unpack_fp4_70[5]), "=f"(_unpack_fp4_70[6]), "=f"(_unpack_fp4_70[7]) : "r"(_fp4_wire_values_1[10 + 0]));
                        #endif
                    }
                    float2 a_152 = make_float2(acc_125[16], acc_125[17]);
                    float2 q_153 = make_float2(_unpack_fp4_70[0], _unpack_fp4_70[1]);
                    float2 r_154 = fma_f32x2_rn_noftz(q_153, scale1_127, a_152);
                    acc_125[16] = r_154.x;
                    acc_125[17] = r_154.y;
                    float2 a_155 = make_float2(acc_125[18], acc_125[19]);
                    float2 q_156 = make_float2(_unpack_fp4_70[2], _unpack_fp4_70[3]);
                    float2 r_157 = fma_f32x2_rn_noftz(q_156, scale1_127, a_155);
                    acc_125[18] = r_157.x;
                    acc_125[19] = r_157.y;
                    float2 a_158_1 = make_float2(acc_125[20], acc_125[21]);
                    float2 q_159_1 = make_float2(_unpack_fp4_70[4], _unpack_fp4_70[5]);
                    float2 r_160_1 = fma_f32x2_rn_noftz(q_159_1, scale1_127, a_158_1);
                    acc_125[20] = r_160_1.x;
                    acc_125[21] = r_160_1.y;
                    float2 a_161_1 = make_float2(acc_125[22], acc_125[23]);
                    float2 q_162_1 = make_float2(_unpack_fp4_70[6], _unpack_fp4_70[7]);
                    float2 r_163_1 = fma_f32x2_rn_noftz(q_162_1, scale1_127, a_161_1);
                    acc_125[22] = r_163_1.x;
                    acc_125[23] = r_163_1.y;
                    float _unpack_fp4_71[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(_fp4_wire_values_1[11 + 0]));
                        (&_unpack_fp4_71[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_71[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_71[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_71[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_71[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_71[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_71[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_71[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_71[0]), "=f"(_unpack_fp4_71[1]), "=f"(_unpack_fp4_71[2]), "=f"(_unpack_fp4_71[3]), "=f"(_unpack_fp4_71[4]), "=f"(_unpack_fp4_71[5]), "=f"(_unpack_fp4_71[6]), "=f"(_unpack_fp4_71[7]) : "r"(_fp4_wire_values_1[11 + 0]));
                        #endif
                    }
                    float2 a_164_1 = make_float2(acc_125[24], acc_125[25]);
                    float2 q_165_1 = make_float2(_unpack_fp4_71[0], _unpack_fp4_71[1]);
                    float2 r_166_1 = fma_f32x2_rn_noftz(q_165_1, scale1_127, a_164_1);
                    acc_125[24] = r_166_1.x;
                    acc_125[25] = r_166_1.y;
                    float2 a_167_1 = make_float2(acc_125[26], acc_125[27]);
                    float2 q_168_1 = make_float2(_unpack_fp4_71[2], _unpack_fp4_71[3]);
                    float2 r_169_1 = fma_f32x2_rn_noftz(q_168_1, scale1_127, a_167_1);
                    acc_125[26] = r_169_1.x;
                    acc_125[27] = r_169_1.y;
                    float2 a_170_1 = make_float2(acc_125[28], acc_125[29]);
                    float2 q_171_1 = make_float2(_unpack_fp4_71[4], _unpack_fp4_71[5]);
                    float2 r_172_1 = fma_f32x2_rn_noftz(q_171_1, scale1_127, a_170_1);
                    acc_125[28] = r_172_1.x;
                    acc_125[29] = r_172_1.y;
                    float2 a_173_1 = make_float2(acc_125[30], acc_125[31]);
                    float2 q_174_1 = make_float2(_unpack_fp4_71[6], _unpack_fp4_71[7]);
                    float2 r_175_1 = fma_f32x2_rn_noftz(q_174_1, scale1_127, a_173_1);
                    acc_125[30] = r_175_1.x;
                    acc_125[31] = r_175_1.y;
                    unsigned int acc_125_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_125[_lp*2 + 0], acc_125[_lp*2+1 + 0]));
                        acc_125_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 0)))), "r"(acc_125_bf16[0]), "r"(acc_125_bf16[1]), "r"(acc_125_bf16[2]), "r"(acc_125_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 1)))), "r"(acc_125_bf16[4]), "r"(acc_125_bf16[5]), "r"(acc_125_bf16[6]), "r"(acc_125_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 2)))), "r"(acc_125_bf16[8]), "r"(acc_125_bf16[9]), "r"(acc_125_bf16[10]), "r"(acc_125_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * lane_1 + (f ^ 3)))), "r"(acc_125_bf16[12]), "r"(acc_125_bf16[13]), "r"(acc_125_bf16[14]), "r"(acc_125_bf16[15]) : "memory");
                    __syncwarp();
                    int c_176 = lane_1;
                    int l_177 = c_176 >> 2;
                    unsigned int w4_178[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_178[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_178[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_177 + (c_176 & 3 ^ l_177 >> 1 & 3)))));
                    long long idx_179 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_176));
                    {
                        int4 _iv4 = make_int4(w4_178[0 + 0], w4_178[0 + 1], w4_178[0 + 2], w4_178[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_179) = _iv4;
                    }
                    int c_180 = lane_1 + 32;
                    int l_181 = c_180 >> 2;
                    unsigned int w4_182[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_182[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_182[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_181 + (c_180 & 3 ^ l_181 >> 1 & 3)))));
                    long long idx_183 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_180));
                    {
                        int4 _iv4 = make_int4(w4_182[0 + 0], w4_182[0 + 1], w4_182[0 + 2], w4_182[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_183) = _iv4;
                    }
                    int c_184 = lane_1 + 64;
                    int l_185 = c_184 >> 2;
                    unsigned int w4_186[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_186[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_186[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_185 + (c_184 & 3 ^ l_185 >> 1 & 3)))));
                    long long idx_187 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_184));
                    {
                        int4 _iv4 = make_int4(w4_186[0 + 0], w4_186[0 + 1], w4_186[0 + 2], w4_186[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_187) = _iv4;
                    }
                    int c_188 = lane_1 + 96;
                    int l_189 = c_188 >> 2;
                    unsigned int w4_190[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_190[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_190[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(stg_off + 16 * (4 * l_189 + (c_188 & 3 ^ l_189 >> 1 & 3)))));
                    long long idx_191 = crow64 * 3072 + (long long)(4 * (128 * (3 * jj + 2) + c_188));
                    {
                        int4 _iv4 = make_int4(w4_190[0 + 0], w4_190[0 + 1], w4_190[0 + 2], w4_190[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx_191) = _iv4;
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
            int g32_1 = (int)gridDim.x;
            if (rows / g32_1 <= 8) {
                asm volatile("barrier.sync 7, 352;" ::: "memory");
            }
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
