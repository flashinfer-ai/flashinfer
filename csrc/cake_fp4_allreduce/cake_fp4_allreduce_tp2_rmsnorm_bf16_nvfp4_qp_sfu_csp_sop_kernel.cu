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
#define SMEM_WROW_OFF 1024
#define SMEM_WROW_STAGE_BYTES 12288
#define SMEM_WROW_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 212800
#define SMEM_WSM_STRIDE 212800
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 212800
#define SMEM_WSM32_STRIDE 212800
#define SMEM_TOTAL 226176
#define THREADS 352
#define LAUNCH_MIN_BLOCKS 1
#ifndef CAKE_FP4_ALLREDUCE_KERNEL
#define CAKE_FP4_ALLREDUCE_KERNEL cake_fp4_allreduce_tp2_rmsnorm_bf16_nvfp4_qp_sfu_csp_sop
#endif

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


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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
CAKE_FP4_ALLREDUCE_KERNEL(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ outq, uint8_t* __restrict__ out_sf, float* __restrict__ gs_out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
    #define rb_full_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* wrow = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int wrow_addr = smem + 1024;
    unsigned int* ctl = reinterpret_cast<unsigned int*>(smem_raw + 13312);
    const int ctl_addr = smem + 13312;
    uint8_t* wsm = reinterpret_cast<uint8_t*>(smem_raw + 13376);
    const int wsm_addr = smem + 13376;
    float* wsm32 = reinterpret_cast<float*>(smem_raw + 13376);
    const int wsm32_addr = smem + 13376;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

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
            // rb_full: 10 barriers, init_count=1
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
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 320) {
                unsigned int _vec_load_0[4];
                {
                    uint32_t _uv4_0_0;
                    uint32_t _uv4_0_1;
                    uint32_t _uv4_0_2;
                    uint32_t _uv4_0_3;
                    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(w) + (16 * i)) + (0))) : "memory");
                    _vec_load_0[0 + 0] = _uv4_0_0;
                    _vec_load_0[0 + 1] = _uv4_0_1;
                    _vec_load_0[0 + 2] = _uv4_0_2;
                    _vec_load_0[0 + 3] = _uv4_0_3;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow_addr + (unsigned int)(16 * i)), "r"(_vec_load_0[0]), "r"(_vec_load_0[1]), "r"(_vec_load_0[2]), "r"(_vec_load_0[3]) : "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            long long bid64 = (long long)bid;
            long long g64 = (long long)gridDim.x;
            long long rows64 = (long long)rows;
            int r0 = (int)(bid64 * rows64 / g64);
            int r1 = (int)((bid64 + 1) * rows64 / g64);
            int nrows = r1 - r0;
            int warp_1 = (int)warp;
            int lane_2 = (int)lane;
            int n = 0;
            if (nrows > warp_1) {
                n = (nrows - warp_1 + 10 - 1) / 10;
            }
            if (n > 0 && lane_2 == 0) {
                long long first_row = (long long)(r0 + warp_1);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 21280), x + (first_row * 6144), 12288, xb_full_addr + (warp_1) * 8);
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
            int f = lane_2 >> 1 & 3;
            int xb_off = warp_1 * 21280;
            int own_off = xb_off + 12288;
            int stg_off = own_off + 6944;
            int rb_off = xb_off;
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
            float gso = gs_out[0];
            float gs_me = 1.0f;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 10 * k);
                    int ob_off = own_off + k % 2 * 3472;
#ifdef CAKE_FP4_ALLREDUCE_PREFETCH
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * lane_2)))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 32))))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 64))))));
#endif
                    if (k > 1) {
                        if (lane_2 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncwarp();
                    }
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (warp_1) * 8, k & 1);
                    unsigned int bmp[6];
                    float am = 0.0f;
                    int pp = lane_2;
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
                    unsigned int _max_1 = ((xa) > (xb) ? (xa) : (xb));
                    am = fmaxf(am, __uint_as_float(_max_1 << 16));
                    int pp_0 = lane_2 + 32;
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
                    unsigned int _max_2 = ((xa_8) > (xb_9) ? (xa_8) : (xb_9));
                    am = fmaxf(am, __uint_as_float(_max_2 << 16));
                    int pp_10 = lane_2 + 64;
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
                    unsigned int _max_3 = ((xa_18) > (xb_19) ? (xa_18) : (xb_19));
                    am = fmaxf(am, __uint_as_float(_max_3 << 16));
                    int pp_20 = lane_2 + 96;
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
                    unsigned int _max_4 = ((xa_28) > (xb_29) ? (xa_28) : (xb_29));
                    am = fmaxf(am, __uint_as_float(_max_4 << 16));
                    int pp_30 = lane_2 + 128;
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
                    unsigned int _max_5 = ((xa_38) > (xb_39) ? (xa_38) : (xb_39));
                    am = fmaxf(am, __uint_as_float(_max_5 << 16));
                    int pp_40 = lane_2 + 160;
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
                    unsigned int _max_6 = ((xa_48) > (xb_49) ? (xa_48) : (xb_49));
                    am = fmaxf(am, __uint_as_float(_max_6 << 16));
                    float _warp_reduce_7 = am;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_7 = max_noftz(_warp_reduce_7, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset));
                    am = _warp_reduce_7;
                    float _fdiv_rn_0 = __fdiv_rn(2688.0f, am);
                    float inv = 0.0f;
                    if (am > 0.0f) {
                        inv = _fdiv_rn_0;
                    }
                    float rs_50 = __fdiv_rn(am, 2688.0f);
                    int pp2 = lane_2;
                    int base2 = xb_off + 64 * pp2;
                    float bm0 = __uint_as_float(bmp[0] << 16);
                    float bm1 = __uint_as_float(bmp[0] & 4294901760u);
                    float _fdiv_8 = __fdiv_rn(mul_f32_rn_noftz(bm1, inv), 6.0f);
                    float _fdiv_9 = __fdiv_rn(mul_f32_rn_noftz(bm0, inv), 6.0f);
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_fdiv_8), "f"(_fdiv_9));
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
                    unsigned int w_51[4];
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
                    uint16_t _e2m1_pair_h_10;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g.x), "f"(g.y));
                    _fp4_pair_0 = (uint32_t)_e2m1_pair_h_10;
                    float2 pair_52 = make_float2(__uint_as_float(v4[1] << 16), __uint_as_float(v4[1] & 4294901760u));
                    float2 g_53;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_53) : "l"(*(const unsigned long long*)&pair_52), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_1;
                    uint16_t _e2m1_pair_h_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_53.x), "f"(g_53.y));
                    _fp4_pair_1 = (uint32_t)_e2m1_pair_h_11;
                    float2 pair_54 = make_float2(__uint_as_float(v4[2] << 16), __uint_as_float(v4[2] & 4294901760u));
                    float2 g_55;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_55) : "l"(*(const unsigned long long*)&pair_54), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_pair_2;
                    uint16_t _e2m1_pair_h_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_55.x), "f"(g_55.y));
                    _fp4_pair_2 = (uint32_t)_e2m1_pair_h_12;
                    float2 pair_56 = make_float2(__uint_as_float(v4[3] << 16), __uint_as_float(v4[3] & 4294901760u));
                    float2 g_57;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_57) : "l"(*(const unsigned long long*)&pair_56), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_e2m1_pack_pair_13;
                    uint16_t _e2m1_pair_h_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_57.x), "f"(g_57.y));
                    _fp4_e2m1_pack_pair_13 = (uint32_t)_e2m1_pair_h_14;
                    unsigned int word = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_13 << 24;
                    unsigned int carry = (word & 2004318071) + 2004318071;
                    w_51[0] = word & (carry | 2004318071);
                    int cf_58 = f ^ 1;
                    unsigned int v4_59[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_59[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_59[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_59[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_59[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_58)));
                    float m_60 = m0;
                    if ((cf_58 & 2) != 0) {
                        m_60 = m1;
                    }
                    float2 mm_61 = make_float2(m_60, m_60);
                    float2 pair_62 = make_float2(__uint_as_float(v4_59[0] << 16), __uint_as_float(v4_59[0] & 4294901760u));
                    float2 g_63;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_63) : "l"(*(const unsigned long long*)&pair_62), "l"(*(const unsigned long long*)&mm_61));
                    uint32_t _fp4_pair_3;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_63.x), "f"(g_63.y));
                    _fp4_pair_3 = (uint32_t)_e2m1_pair_h_15;
                    float2 pair_64 = make_float2(__uint_as_float(v4_59[1] << 16), __uint_as_float(v4_59[1] & 4294901760u));
                    float2 g_65;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_65) : "l"(*(const unsigned long long*)&pair_64), "l"(*(const unsigned long long*)&mm_61));
                    uint32_t _fp4_pair_4;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_65.x), "f"(g_65.y));
                    _fp4_pair_4 = (uint32_t)_e2m1_pair_h_16;
                    float2 pair_66 = make_float2(__uint_as_float(v4_59[2] << 16), __uint_as_float(v4_59[2] & 4294901760u));
                    float2 g_67;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_67) : "l"(*(const unsigned long long*)&pair_66), "l"(*(const unsigned long long*)&mm_61));
                    uint32_t _fp4_pair_5;
                    uint16_t _e2m1_pair_h_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_67.x), "f"(g_67.y));
                    _fp4_pair_5 = (uint32_t)_e2m1_pair_h_17;
                    float2 pair_68 = make_float2(__uint_as_float(v4_59[3] << 16), __uint_as_float(v4_59[3] & 4294901760u));
                    float2 g_69;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_69) : "l"(*(const unsigned long long*)&pair_68), "l"(*(const unsigned long long*)&mm_61));
                    uint32_t _fp4_e2m1_pack_pair_18;
                    uint16_t _e2m1_pair_h_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_69.x), "f"(g_69.y));
                    _fp4_e2m1_pack_pair_18 = (uint32_t)_e2m1_pair_h_19;
                    unsigned int word_70 = _fp4_pair_3 | _fp4_pair_4 << 8 | _fp4_pair_5 << 16 | _fp4_e2m1_pack_pair_18 << 24;
                    unsigned int carry_71 = (word_70 & 2004318071) + 2004318071;
                    w_51[1] = word_70 & (carry_71 | 2004318071);
                    int cf_72 = f ^ 2;
                    unsigned int v4_73[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_73[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_73[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_73[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_73[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_72)));
                    float m_74 = m0;
                    if ((cf_72 & 2) != 0) {
                        m_74 = m1;
                    }
                    float2 mm_75 = make_float2(m_74, m_74);
                    float2 pair_76 = make_float2(__uint_as_float(v4_73[0] << 16), __uint_as_float(v4_73[0] & 4294901760u));
                    float2 g_77;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_77) : "l"(*(const unsigned long long*)&pair_76), "l"(*(const unsigned long long*)&mm_75));
                    uint32_t _fp4_pair_6;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_77.x), "f"(g_77.y));
                    _fp4_pair_6 = (uint32_t)_e2m1_pair_h_20;
                    float2 pair_78 = make_float2(__uint_as_float(v4_73[1] << 16), __uint_as_float(v4_73[1] & 4294901760u));
                    float2 g_79;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_79) : "l"(*(const unsigned long long*)&pair_78), "l"(*(const unsigned long long*)&mm_75));
                    uint32_t _fp4_pair_7;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_79.x), "f"(g_79.y));
                    _fp4_pair_7 = (uint32_t)_e2m1_pair_h_21;
                    float2 pair_80 = make_float2(__uint_as_float(v4_73[2] << 16), __uint_as_float(v4_73[2] & 4294901760u));
                    float2 g_81;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_81) : "l"(*(const unsigned long long*)&pair_80), "l"(*(const unsigned long long*)&mm_75));
                    uint32_t _fp4_pair_8;
                    uint16_t _e2m1_pair_h_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_81.x), "f"(g_81.y));
                    _fp4_pair_8 = (uint32_t)_e2m1_pair_h_22;
                    float2 pair_82 = make_float2(__uint_as_float(v4_73[3] << 16), __uint_as_float(v4_73[3] & 4294901760u));
                    float2 g_83;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_83) : "l"(*(const unsigned long long*)&pair_82), "l"(*(const unsigned long long*)&mm_75));
                    uint32_t _fp4_e2m1_pack_pair_23;
                    uint16_t _e2m1_pair_h_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_83.x), "f"(g_83.y));
                    _fp4_e2m1_pack_pair_23 = (uint32_t)_e2m1_pair_h_24;
                    unsigned int word_84 = _fp4_pair_6 | _fp4_pair_7 << 8 | _fp4_pair_8 << 16 | _fp4_e2m1_pack_pair_23 << 24;
                    unsigned int carry_85 = (word_84 & 2004318071) + 2004318071;
                    w_51[2] = word_84 & (carry_85 | 2004318071);
                    int cf_86 = f ^ 3;
                    unsigned int v4_87[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_87[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_87[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_87[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_87[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2 + 16 * cf_86)));
                    float m_88 = m0;
                    if ((cf_86 & 2) != 0) {
                        m_88 = m1;
                    }
                    float2 mm_89 = make_float2(m_88, m_88);
                    float2 pair_90 = make_float2(__uint_as_float(v4_87[0] << 16), __uint_as_float(v4_87[0] & 4294901760u));
                    float2 g_91;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_91) : "l"(*(const unsigned long long*)&pair_90), "l"(*(const unsigned long long*)&mm_89));
                    uint32_t _fp4_pair_9;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_91.x), "f"(g_91.y));
                    _fp4_pair_9 = (uint32_t)_e2m1_pair_h_25;
                    float2 pair_92 = make_float2(__uint_as_float(v4_87[1] << 16), __uint_as_float(v4_87[1] & 4294901760u));
                    float2 g_93;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_93) : "l"(*(const unsigned long long*)&pair_92), "l"(*(const unsigned long long*)&mm_89));
                    uint32_t _fp4_pair_10;
                    uint16_t _e2m1_pair_h_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_93.x), "f"(g_93.y));
                    _fp4_pair_10 = (uint32_t)_e2m1_pair_h_26;
                    float2 pair_94 = make_float2(__uint_as_float(v4_87[2] << 16), __uint_as_float(v4_87[2] & 4294901760u));
                    float2 g_95;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_95) : "l"(*(const unsigned long long*)&pair_94), "l"(*(const unsigned long long*)&mm_89));
                    uint32_t _fp4_pair_11;
                    uint16_t _e2m1_pair_h_27;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(g_95.x), "f"(g_95.y));
                    _fp4_pair_11 = (uint32_t)_e2m1_pair_h_27;
                    float2 pair_96 = make_float2(__uint_as_float(v4_87[3] << 16), __uint_as_float(v4_87[3] & 4294901760u));
                    float2 g_97;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_97) : "l"(*(const unsigned long long*)&pair_96), "l"(*(const unsigned long long*)&mm_89));
                    uint32_t _fp4_e2m1_pack_pair_28;
                    uint16_t _e2m1_pair_h_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_97.x), "f"(g_97.y));
                    _fp4_e2m1_pack_pair_28 = (uint32_t)_e2m1_pair_h_29;
                    unsigned int word_98 = _fp4_pair_9 | _fp4_pair_10 << 8 | _fp4_pair_11 << 16 | _fp4_e2m1_pack_pair_28 << 24;
                    unsigned int carry_99 = (word_98 & 2004318071) + 2004318071;
                    w_51[3] = word_98 & (carry_99 | 2004318071);
                    bool fodd = (f & 1) != 0;
                    bool fhi = (f & 2) != 0;
                    unsigned int t0 = w_51[0];
                    unsigned int t1 = w_51[1];
                    unsigned int t2 = w_51[2];
                    unsigned int t3 = w_51[3];
                    if (fodd) {
                        t0 = w_51[1];
                        t1 = w_51[0];
                        t2 = w_51[3];
                        t3 = w_51[2];
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
                    int pp2_100 = lane_2 + 32;
                    int base2_101 = xb_off + 64 * pp2_100;
                    float bm0_102 = __uint_as_float(bmp[1] << 16);
                    float bm1_103 = __uint_as_float(bmp[1] & 4294901760u);
                    float _fdiv_30 = __fdiv_rn(mul_f32_rn_noftz(bm1_103, inv), 6.0f);
                    float _fdiv_31 = __fdiv_rn(mul_f32_rn_noftz(bm0_102, inv), 6.0f);
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_fdiv_30), "f"(_fdiv_31));
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(_e4m3x2_f32_1)));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _fdiv_rn_3 = __fdiv_rn(inv, _cvt_f32_f16_2);
                    float _fdiv_rn_4 = __fdiv_rn(inv, _cvt_f32_f16_3);
                    float m0_104 = 0.0f;
                    float m1_105 = 0.0f;
                    if (_cvt_f32_f16_2 > 0.0f) {
                        m0_104 = _fdiv_rn_3;
                    }
                    if (_cvt_f32_f16_3 > 0.0f) {
                        m1_105 = _fdiv_rn_4;
                    }
                    unsigned int w_106[4];
                    int cf_107 = f ^ 0;
                    unsigned int v4_108[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_108[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_108[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_108[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_108[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_101 + 16 * cf_107)));
                    float m_109 = m0_104;
                    if ((cf_107 & 2) != 0) {
                        m_109 = m1_105;
                    }
                    float2 mm_110 = make_float2(m_109, m_109);
                    float2 pair_111 = make_float2(__uint_as_float(v4_108[0] << 16), __uint_as_float(v4_108[0] & 4294901760u));
                    float2 g_112;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_112) : "l"(*(const unsigned long long*)&pair_111), "l"(*(const unsigned long long*)&mm_110));
                    uint32_t _fp4_pair_12;
                    uint16_t _e2m1_pair_h_32;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_112.x), "f"(g_112.y));
                    _fp4_pair_12 = (uint32_t)_e2m1_pair_h_32;
                    float2 pair_113 = make_float2(__uint_as_float(v4_108[1] << 16), __uint_as_float(v4_108[1] & 4294901760u));
                    float2 g_114;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_114) : "l"(*(const unsigned long long*)&pair_113), "l"(*(const unsigned long long*)&mm_110));
                    uint32_t _fp4_pair_13;
                    uint16_t _e2m1_pair_h_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_114.x), "f"(g_114.y));
                    _fp4_pair_13 = (uint32_t)_e2m1_pair_h_33;
                    float2 pair_115 = make_float2(__uint_as_float(v4_108[2] << 16), __uint_as_float(v4_108[2] & 4294901760u));
                    float2 g_116;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_116) : "l"(*(const unsigned long long*)&pair_115), "l"(*(const unsigned long long*)&mm_110));
                    uint32_t _fp4_pair_14;
                    uint16_t _e2m1_pair_h_34;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(g_116.x), "f"(g_116.y));
                    _fp4_pair_14 = (uint32_t)_e2m1_pair_h_34;
                    float2 pair_117 = make_float2(__uint_as_float(v4_108[3] << 16), __uint_as_float(v4_108[3] & 4294901760u));
                    float2 g_118;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_118) : "l"(*(const unsigned long long*)&pair_117), "l"(*(const unsigned long long*)&mm_110));
                    uint32_t _fp4_e2m1_pack_pair_35;
                    uint16_t _e2m1_pair_h_36;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(g_118.x), "f"(g_118.y));
                    _fp4_e2m1_pack_pair_35 = (uint32_t)_e2m1_pair_h_36;
                    unsigned int word_119 = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_e2m1_pack_pair_35 << 24;
                    unsigned int carry_120 = (word_119 & 2004318071) + 2004318071;
                    w_106[0] = word_119 & (carry_120 | 2004318071);
                    int cf_121 = f ^ 1;
                    unsigned int v4_122[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_122[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_122[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_122[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_122[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_101 + 16 * cf_121)));
                    float m_123 = m0_104;
                    if ((cf_121 & 2) != 0) {
                        m_123 = m1_105;
                    }
                    float2 mm_124 = make_float2(m_123, m_123);
                    float2 pair_125 = make_float2(__uint_as_float(v4_122[0] << 16), __uint_as_float(v4_122[0] & 4294901760u));
                    float2 g_126;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_126) : "l"(*(const unsigned long long*)&pair_125), "l"(*(const unsigned long long*)&mm_124));
                    uint32_t _fp4_pair_15;
                    uint16_t _e2m1_pair_h_37;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_126.x), "f"(g_126.y));
                    _fp4_pair_15 = (uint32_t)_e2m1_pair_h_37;
                    float2 pair_127 = make_float2(__uint_as_float(v4_122[1] << 16), __uint_as_float(v4_122[1] & 4294901760u));
                    float2 g_128;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_128) : "l"(*(const unsigned long long*)&pair_127), "l"(*(const unsigned long long*)&mm_124));
                    uint32_t _fp4_pair_16;
                    uint16_t _e2m1_pair_h_38;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_128.x), "f"(g_128.y));
                    _fp4_pair_16 = (uint32_t)_e2m1_pair_h_38;
                    float2 pair_129 = make_float2(__uint_as_float(v4_122[2] << 16), __uint_as_float(v4_122[2] & 4294901760u));
                    float2 g_130;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_130) : "l"(*(const unsigned long long*)&pair_129), "l"(*(const unsigned long long*)&mm_124));
                    uint32_t _fp4_pair_17;
                    uint16_t _e2m1_pair_h_39;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_130.x), "f"(g_130.y));
                    _fp4_pair_17 = (uint32_t)_e2m1_pair_h_39;
                    float2 pair_131 = make_float2(__uint_as_float(v4_122[3] << 16), __uint_as_float(v4_122[3] & 4294901760u));
                    float2 g_132;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_132) : "l"(*(const unsigned long long*)&pair_131), "l"(*(const unsigned long long*)&mm_124));
                    uint32_t _fp4_e2m1_pack_pair_40;
                    uint16_t _e2m1_pair_h_41;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_132.x), "f"(g_132.y));
                    _fp4_e2m1_pack_pair_40 = (uint32_t)_e2m1_pair_h_41;
                    unsigned int word_133 = _fp4_pair_15 | _fp4_pair_16 << 8 | _fp4_pair_17 << 16 | _fp4_e2m1_pack_pair_40 << 24;
                    unsigned int carry_134 = (word_133 & 2004318071) + 2004318071;
                    w_106[1] = word_133 & (carry_134 | 2004318071);
                    int cf_135 = f ^ 2;
                    unsigned int v4_136[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_136[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_136[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_136[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_136[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_101 + 16 * cf_135)));
                    float m_137 = m0_104;
                    if ((cf_135 & 2) != 0) {
                        m_137 = m1_105;
                    }
                    float2 mm_138 = make_float2(m_137, m_137);
                    float2 pair_139 = make_float2(__uint_as_float(v4_136[0] << 16), __uint_as_float(v4_136[0] & 4294901760u));
                    float2 g_140;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_140) : "l"(*(const unsigned long long*)&pair_139), "l"(*(const unsigned long long*)&mm_138));
                    uint32_t _fp4_pair_18;
                    uint16_t _e2m1_pair_h_42;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_140.x), "f"(g_140.y));
                    _fp4_pair_18 = (uint32_t)_e2m1_pair_h_42;
                    float2 pair_141 = make_float2(__uint_as_float(v4_136[1] << 16), __uint_as_float(v4_136[1] & 4294901760u));
                    float2 g_142;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_142) : "l"(*(const unsigned long long*)&pair_141), "l"(*(const unsigned long long*)&mm_138));
                    uint32_t _fp4_pair_19;
                    uint16_t _e2m1_pair_h_43;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_142.x), "f"(g_142.y));
                    _fp4_pair_19 = (uint32_t)_e2m1_pair_h_43;
                    float2 pair_143 = make_float2(__uint_as_float(v4_136[2] << 16), __uint_as_float(v4_136[2] & 4294901760u));
                    float2 g_144;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_144) : "l"(*(const unsigned long long*)&pair_143), "l"(*(const unsigned long long*)&mm_138));
                    uint32_t _fp4_pair_20;
                    uint16_t _e2m1_pair_h_44;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_144.x), "f"(g_144.y));
                    _fp4_pair_20 = (uint32_t)_e2m1_pair_h_44;
                    float2 pair_145 = make_float2(__uint_as_float(v4_136[3] << 16), __uint_as_float(v4_136[3] & 4294901760u));
                    float2 g_146;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_146) : "l"(*(const unsigned long long*)&pair_145), "l"(*(const unsigned long long*)&mm_138));
                    uint32_t _fp4_e2m1_pack_pair_45;
                    uint16_t _e2m1_pair_h_46;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_146.x), "f"(g_146.y));
                    _fp4_e2m1_pack_pair_45 = (uint32_t)_e2m1_pair_h_46;
                    unsigned int word_147 = _fp4_pair_18 | _fp4_pair_19 << 8 | _fp4_pair_20 << 16 | _fp4_e2m1_pack_pair_45 << 24;
                    unsigned int carry_148 = (word_147 & 2004318071) + 2004318071;
                    w_106[2] = word_147 & (carry_148 | 2004318071);
                    int cf_149 = f ^ 3;
                    unsigned int v4_150[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_150[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_150[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_101 + 16 * cf_149)));
                    float m_151 = m0_104;
                    if ((cf_149 & 2) != 0) {
                        m_151 = m1_105;
                    }
                    float2 mm_152 = make_float2(m_151, m_151);
                    float2 pair_153 = make_float2(__uint_as_float(v4_150[0] << 16), __uint_as_float(v4_150[0] & 4294901760u));
                    float2 g_154;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_154) : "l"(*(const unsigned long long*)&pair_153), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_21;
                    uint16_t _e2m1_pair_h_47;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_154.x), "f"(g_154.y));
                    _fp4_pair_21 = (uint32_t)_e2m1_pair_h_47;
                    float2 pair_155 = make_float2(__uint_as_float(v4_150[1] << 16), __uint_as_float(v4_150[1] & 4294901760u));
                    float2 g_156;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_156) : "l"(*(const unsigned long long*)&pair_155), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_22;
                    uint16_t _e2m1_pair_h_48;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(g_156.x), "f"(g_156.y));
                    _fp4_pair_22 = (uint32_t)_e2m1_pair_h_48;
                    float2 pair_157 = make_float2(__uint_as_float(v4_150[2] << 16), __uint_as_float(v4_150[2] & 4294901760u));
                    float2 g_158;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_158) : "l"(*(const unsigned long long*)&pair_157), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_pair_23;
                    uint16_t _e2m1_pair_h_49;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_49) : "f"(g_158.x), "f"(g_158.y));
                    _fp4_pair_23 = (uint32_t)_e2m1_pair_h_49;
                    float2 pair_159 = make_float2(__uint_as_float(v4_150[3] << 16), __uint_as_float(v4_150[3] & 4294901760u));
                    float2 g_160;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_160) : "l"(*(const unsigned long long*)&pair_159), "l"(*(const unsigned long long*)&mm_152));
                    uint32_t _fp4_e2m1_pack_pair_50;
                    uint16_t _e2m1_pair_h_51;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(g_160.x), "f"(g_160.y));
                    _fp4_e2m1_pack_pair_50 = (uint32_t)_e2m1_pair_h_51;
                    unsigned int word_161 = _fp4_pair_21 | _fp4_pair_22 << 8 | _fp4_pair_23 << 16 | _fp4_e2m1_pack_pair_50 << 24;
                    unsigned int carry_162 = (word_161 & 2004318071) + 2004318071;
                    w_106[3] = word_161 & (carry_162 | 2004318071);
                    bool fodd_163 = (f & 1) != 0;
                    bool fhi_164 = (f & 2) != 0;
                    unsigned int t0_165 = w_106[0];
                    unsigned int t1_166 = w_106[1];
                    unsigned int t2_167 = w_106[2];
                    unsigned int t3_168 = w_106[3];
                    if (fodd_163) {
                        t0_165 = w_106[1];
                        t1_166 = w_106[0];
                        t2_167 = w_106[3];
                        t3_168 = w_106[2];
                    }
                    unsigned int q0_169 = t0_165;
                    unsigned int q1_170 = t1_166;
                    unsigned int q2_171 = t2_167;
                    unsigned int q3_172 = t3_168;
                    if (fhi_164) {
                        q0_169 = t2_167;
                        q1_170 = t3_168;
                        q2_171 = t0_165;
                        q3_172 = t1_166;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_100)), "r"(q0_169), "r"(q1_170), "r"(q2_171), "r"(q3_172) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_100)), "h"((uint16_t)(_e4m3x2_f32_1)) : "memory");
                    int pp2_173 = lane_2 + 64;
                    int base2_174 = xb_off + 64 * pp2_173;
                    float bm0_175 = __uint_as_float(bmp[2] << 16);
                    float bm1_176 = __uint_as_float(bmp[2] & 4294901760u);
                    float _fdiv_52 = __fdiv_rn(mul_f32_rn_noftz(bm1_176, inv), 6.0f);
                    float _fdiv_53 = __fdiv_rn(mul_f32_rn_noftz(bm0_175, inv), 6.0f);
                    uint16_t _e4m3x2_f32_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(_fdiv_52), "f"(_fdiv_53));
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(_e4m3x2_f32_2)));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _fdiv_rn_5 = __fdiv_rn(inv, _cvt_f32_f16_4);
                    float _fdiv_rn_6 = __fdiv_rn(inv, _cvt_f32_f16_5);
                    float m0_177 = 0.0f;
                    float m1_178 = 0.0f;
                    if (_cvt_f32_f16_4 > 0.0f) {
                        m0_177 = _fdiv_rn_5;
                    }
                    if (_cvt_f32_f16_5 > 0.0f) {
                        m1_178 = _fdiv_rn_6;
                    }
                    unsigned int w_179[4];
                    int cf_180 = f ^ 0;
                    unsigned int v4_181[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_181[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_181[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_181[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_181[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_174 + 16 * cf_180)));
                    float m_182 = m0_177;
                    if ((cf_180 & 2) != 0) {
                        m_182 = m1_178;
                    }
                    float2 mm_183 = make_float2(m_182, m_182);
                    float2 pair_184 = make_float2(__uint_as_float(v4_181[0] << 16), __uint_as_float(v4_181[0] & 4294901760u));
                    float2 g_185;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_185) : "l"(*(const unsigned long long*)&pair_184), "l"(*(const unsigned long long*)&mm_183));
                    uint32_t _fp4_pair_24;
                    uint16_t _e2m1_pair_h_54;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_185.x), "f"(g_185.y));
                    _fp4_pair_24 = (uint32_t)_e2m1_pair_h_54;
                    float2 pair_186 = make_float2(__uint_as_float(v4_181[1] << 16), __uint_as_float(v4_181[1] & 4294901760u));
                    float2 g_187;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_187) : "l"(*(const unsigned long long*)&pair_186), "l"(*(const unsigned long long*)&mm_183));
                    uint32_t _fp4_pair_25;
                    uint16_t _e2m1_pair_h_55;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_187.x), "f"(g_187.y));
                    _fp4_pair_25 = (uint32_t)_e2m1_pair_h_55;
                    float2 pair_188 = make_float2(__uint_as_float(v4_181[2] << 16), __uint_as_float(v4_181[2] & 4294901760u));
                    float2 g_189;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_189) : "l"(*(const unsigned long long*)&pair_188), "l"(*(const unsigned long long*)&mm_183));
                    uint32_t _fp4_pair_26;
                    uint16_t _e2m1_pair_h_56;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(g_189.x), "f"(g_189.y));
                    _fp4_pair_26 = (uint32_t)_e2m1_pair_h_56;
                    float2 pair_190 = make_float2(__uint_as_float(v4_181[3] << 16), __uint_as_float(v4_181[3] & 4294901760u));
                    float2 g_191;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_191) : "l"(*(const unsigned long long*)&pair_190), "l"(*(const unsigned long long*)&mm_183));
                    uint32_t _fp4_e2m1_pack_pair_57;
                    uint16_t _e2m1_pair_h_58;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(g_191.x), "f"(g_191.y));
                    _fp4_e2m1_pack_pair_57 = (uint32_t)_e2m1_pair_h_58;
                    unsigned int word_192 = _fp4_pair_24 | _fp4_pair_25 << 8 | _fp4_pair_26 << 16 | _fp4_e2m1_pack_pair_57 << 24;
                    unsigned int carry_193 = (word_192 & 2004318071) + 2004318071;
                    w_179[0] = word_192 & (carry_193 | 2004318071);
                    int cf_194 = f ^ 1;
                    unsigned int v4_195[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_195[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_195[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_195[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_195[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_174 + 16 * cf_194)));
                    float m_196 = m0_177;
                    if ((cf_194 & 2) != 0) {
                        m_196 = m1_178;
                    }
                    float2 mm_197 = make_float2(m_196, m_196);
                    float2 pair_198 = make_float2(__uint_as_float(v4_195[0] << 16), __uint_as_float(v4_195[0] & 4294901760u));
                    float2 g_199;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_199) : "l"(*(const unsigned long long*)&pair_198), "l"(*(const unsigned long long*)&mm_197));
                    uint32_t _fp4_pair_27;
                    uint16_t _e2m1_pair_h_59;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_199.x), "f"(g_199.y));
                    _fp4_pair_27 = (uint32_t)_e2m1_pair_h_59;
                    float2 pair_200 = make_float2(__uint_as_float(v4_195[1] << 16), __uint_as_float(v4_195[1] & 4294901760u));
                    float2 g_201;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_201) : "l"(*(const unsigned long long*)&pair_200), "l"(*(const unsigned long long*)&mm_197));
                    uint32_t _fp4_pair_28;
                    uint16_t _e2m1_pair_h_60;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_201.x), "f"(g_201.y));
                    _fp4_pair_28 = (uint32_t)_e2m1_pair_h_60;
                    float2 pair_202 = make_float2(__uint_as_float(v4_195[2] << 16), __uint_as_float(v4_195[2] & 4294901760u));
                    float2 g_203;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_203) : "l"(*(const unsigned long long*)&pair_202), "l"(*(const unsigned long long*)&mm_197));
                    uint32_t _fp4_pair_29;
                    uint16_t _e2m1_pair_h_61;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(g_203.x), "f"(g_203.y));
                    _fp4_pair_29 = (uint32_t)_e2m1_pair_h_61;
                    float2 pair_204 = make_float2(__uint_as_float(v4_195[3] << 16), __uint_as_float(v4_195[3] & 4294901760u));
                    float2 g_205;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_205) : "l"(*(const unsigned long long*)&pair_204), "l"(*(const unsigned long long*)&mm_197));
                    uint32_t _fp4_e2m1_pack_pair_62;
                    uint16_t _e2m1_pair_h_63;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(g_205.x), "f"(g_205.y));
                    _fp4_e2m1_pack_pair_62 = (uint32_t)_e2m1_pair_h_63;
                    unsigned int word_206 = _fp4_pair_27 | _fp4_pair_28 << 8 | _fp4_pair_29 << 16 | _fp4_e2m1_pack_pair_62 << 24;
                    unsigned int carry_207 = (word_206 & 2004318071) + 2004318071;
                    w_179[1] = word_206 & (carry_207 | 2004318071);
                    int cf_208 = f ^ 2;
                    unsigned int v4_209[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_209[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_209[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_209[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_209[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_174 + 16 * cf_208)));
                    float m_210 = m0_177;
                    if ((cf_208 & 2) != 0) {
                        m_210 = m1_178;
                    }
                    float2 mm_211 = make_float2(m_210, m_210);
                    float2 pair_212 = make_float2(__uint_as_float(v4_209[0] << 16), __uint_as_float(v4_209[0] & 4294901760u));
                    float2 g_213;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_213) : "l"(*(const unsigned long long*)&pair_212), "l"(*(const unsigned long long*)&mm_211));
                    uint32_t _fp4_pair_30;
                    uint16_t _e2m1_pair_h_64;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_213.x), "f"(g_213.y));
                    _fp4_pair_30 = (uint32_t)_e2m1_pair_h_64;
                    float2 pair_214 = make_float2(__uint_as_float(v4_209[1] << 16), __uint_as_float(v4_209[1] & 4294901760u));
                    float2 g_215;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_215) : "l"(*(const unsigned long long*)&pair_214), "l"(*(const unsigned long long*)&mm_211));
                    uint32_t _fp4_pair_31;
                    uint16_t _e2m1_pair_h_65;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_215.x), "f"(g_215.y));
                    _fp4_pair_31 = (uint32_t)_e2m1_pair_h_65;
                    float2 pair_216 = make_float2(__uint_as_float(v4_209[2] << 16), __uint_as_float(v4_209[2] & 4294901760u));
                    float2 g_217;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_217) : "l"(*(const unsigned long long*)&pair_216), "l"(*(const unsigned long long*)&mm_211));
                    uint32_t _fp4_pair_32;
                    uint16_t _e2m1_pair_h_66;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_217.x), "f"(g_217.y));
                    _fp4_pair_32 = (uint32_t)_e2m1_pair_h_66;
                    float2 pair_218 = make_float2(__uint_as_float(v4_209[3] << 16), __uint_as_float(v4_209[3] & 4294901760u));
                    float2 g_219;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_219) : "l"(*(const unsigned long long*)&pair_218), "l"(*(const unsigned long long*)&mm_211));
                    uint32_t _fp4_e2m1_pack_pair_67;
                    uint16_t _e2m1_pair_h_68;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_219.x), "f"(g_219.y));
                    _fp4_e2m1_pack_pair_67 = (uint32_t)_e2m1_pair_h_68;
                    unsigned int word_220 = _fp4_pair_30 | _fp4_pair_31 << 8 | _fp4_pair_32 << 16 | _fp4_e2m1_pack_pair_67 << 24;
                    unsigned int carry_221 = (word_220 & 2004318071) + 2004318071;
                    w_179[2] = word_220 & (carry_221 | 2004318071);
                    int cf_222 = f ^ 3;
                    unsigned int v4_223[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_223[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_223[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_223[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_223[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_174 + 16 * cf_222)));
                    float m_224 = m0_177;
                    if ((cf_222 & 2) != 0) {
                        m_224 = m1_178;
                    }
                    float2 mm_225 = make_float2(m_224, m_224);
                    float2 pair_226 = make_float2(__uint_as_float(v4_223[0] << 16), __uint_as_float(v4_223[0] & 4294901760u));
                    float2 g_227;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_227) : "l"(*(const unsigned long long*)&pair_226), "l"(*(const unsigned long long*)&mm_225));
                    uint32_t _fp4_pair_33;
                    uint16_t _e2m1_pair_h_69;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_227.x), "f"(g_227.y));
                    _fp4_pair_33 = (uint32_t)_e2m1_pair_h_69;
                    float2 pair_228 = make_float2(__uint_as_float(v4_223[1] << 16), __uint_as_float(v4_223[1] & 4294901760u));
                    float2 g_229;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_229) : "l"(*(const unsigned long long*)&pair_228), "l"(*(const unsigned long long*)&mm_225));
                    uint32_t _fp4_pair_34;
                    uint16_t _e2m1_pair_h_70;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(g_229.x), "f"(g_229.y));
                    _fp4_pair_34 = (uint32_t)_e2m1_pair_h_70;
                    float2 pair_230 = make_float2(__uint_as_float(v4_223[2] << 16), __uint_as_float(v4_223[2] & 4294901760u));
                    float2 g_231;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_231) : "l"(*(const unsigned long long*)&pair_230), "l"(*(const unsigned long long*)&mm_225));
                    uint32_t _fp4_pair_35;
                    uint16_t _e2m1_pair_h_71;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_71) : "f"(g_231.x), "f"(g_231.y));
                    _fp4_pair_35 = (uint32_t)_e2m1_pair_h_71;
                    float2 pair_232 = make_float2(__uint_as_float(v4_223[3] << 16), __uint_as_float(v4_223[3] & 4294901760u));
                    float2 g_233;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_233) : "l"(*(const unsigned long long*)&pair_232), "l"(*(const unsigned long long*)&mm_225));
                    uint32_t _fp4_e2m1_pack_pair_72;
                    uint16_t _e2m1_pair_h_73;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_73) : "f"(g_233.x), "f"(g_233.y));
                    _fp4_e2m1_pack_pair_72 = (uint32_t)_e2m1_pair_h_73;
                    unsigned int word_234 = _fp4_pair_33 | _fp4_pair_34 << 8 | _fp4_pair_35 << 16 | _fp4_e2m1_pack_pair_72 << 24;
                    unsigned int carry_235 = (word_234 & 2004318071) + 2004318071;
                    w_179[3] = word_234 & (carry_235 | 2004318071);
                    bool fodd_236 = (f & 1) != 0;
                    bool fhi_237 = (f & 2) != 0;
                    unsigned int t0_238 = w_179[0];
                    unsigned int t1_239 = w_179[1];
                    unsigned int t2_240 = w_179[2];
                    unsigned int t3_241 = w_179[3];
                    if (fodd_236) {
                        t0_238 = w_179[1];
                        t1_239 = w_179[0];
                        t2_240 = w_179[3];
                        t3_241 = w_179[2];
                    }
                    unsigned int q0_242 = t0_238;
                    unsigned int q1_243 = t1_239;
                    unsigned int q2_244 = t2_240;
                    unsigned int q3_245 = t3_241;
                    if (fhi_237) {
                        q0_242 = t2_240;
                        q1_243 = t3_241;
                        q2_244 = t0_238;
                        q3_245 = t1_239;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_173)), "r"(q0_242), "r"(q1_243), "r"(q2_244), "r"(q3_245) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_173)), "h"((uint16_t)(_e4m3x2_f32_2)) : "memory");
                    int pp2_246 = lane_2 + 96;
                    int base2_247 = xb_off + 64 * pp2_246;
                    float bm0_248 = __uint_as_float(bmp[3] << 16);
                    float bm1_249 = __uint_as_float(bmp[3] & 4294901760u);
                    float _fdiv_74 = __fdiv_rn(mul_f32_rn_noftz(bm1_249, inv), 6.0f);
                    float _fdiv_75 = __fdiv_rn(mul_f32_rn_noftz(bm0_248, inv), 6.0f);
                    uint16_t _e4m3x2_f32_3;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(_fdiv_74), "f"(_fdiv_75));
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(_e4m3x2_f32_3)));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _fdiv_rn_7 = __fdiv_rn(inv, _cvt_f32_f16_6);
                    float _fdiv_rn_8 = __fdiv_rn(inv, _cvt_f32_f16_7);
                    float m0_250 = 0.0f;
                    float m1_251 = 0.0f;
                    if (_cvt_f32_f16_6 > 0.0f) {
                        m0_250 = _fdiv_rn_7;
                    }
                    if (_cvt_f32_f16_7 > 0.0f) {
                        m1_251 = _fdiv_rn_8;
                    }
                    unsigned int w_252[4];
                    int cf_253 = f ^ 0;
                    unsigned int v4_254[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_254[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_254[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_254[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_254[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_247 + 16 * cf_253)));
                    float m_255 = m0_250;
                    if ((cf_253 & 2) != 0) {
                        m_255 = m1_251;
                    }
                    float2 mm_256 = make_float2(m_255, m_255);
                    float2 pair_257 = make_float2(__uint_as_float(v4_254[0] << 16), __uint_as_float(v4_254[0] & 4294901760u));
                    float2 g_258;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_258) : "l"(*(const unsigned long long*)&pair_257), "l"(*(const unsigned long long*)&mm_256));
                    uint32_t _fp4_pair_36;
                    uint16_t _e2m1_pair_h_76;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(g_258.x), "f"(g_258.y));
                    _fp4_pair_36 = (uint32_t)_e2m1_pair_h_76;
                    float2 pair_259 = make_float2(__uint_as_float(v4_254[1] << 16), __uint_as_float(v4_254[1] & 4294901760u));
                    float2 g_260;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_260) : "l"(*(const unsigned long long*)&pair_259), "l"(*(const unsigned long long*)&mm_256));
                    uint32_t _fp4_pair_37;
                    uint16_t _e2m1_pair_h_77;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(g_260.x), "f"(g_260.y));
                    _fp4_pair_37 = (uint32_t)_e2m1_pair_h_77;
                    float2 pair_261 = make_float2(__uint_as_float(v4_254[2] << 16), __uint_as_float(v4_254[2] & 4294901760u));
                    float2 g_262;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_262) : "l"(*(const unsigned long long*)&pair_261), "l"(*(const unsigned long long*)&mm_256));
                    uint32_t _fp4_pair_38;
                    uint16_t _e2m1_pair_h_78;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_78) : "f"(g_262.x), "f"(g_262.y));
                    _fp4_pair_38 = (uint32_t)_e2m1_pair_h_78;
                    float2 pair_263 = make_float2(__uint_as_float(v4_254[3] << 16), __uint_as_float(v4_254[3] & 4294901760u));
                    float2 g_264;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_264) : "l"(*(const unsigned long long*)&pair_263), "l"(*(const unsigned long long*)&mm_256));
                    uint32_t _fp4_e2m1_pack_pair_79;
                    uint16_t _e2m1_pair_h_80;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_80) : "f"(g_264.x), "f"(g_264.y));
                    _fp4_e2m1_pack_pair_79 = (uint32_t)_e2m1_pair_h_80;
                    unsigned int word_265 = _fp4_pair_36 | _fp4_pair_37 << 8 | _fp4_pair_38 << 16 | _fp4_e2m1_pack_pair_79 << 24;
                    unsigned int carry_266 = (word_265 & 2004318071) + 2004318071;
                    w_252[0] = word_265 & (carry_266 | 2004318071);
                    int cf_267 = f ^ 1;
                    unsigned int v4_268[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_268[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_268[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_268[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_268[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_247 + 16 * cf_267)));
                    float m_269 = m0_250;
                    if ((cf_267 & 2) != 0) {
                        m_269 = m1_251;
                    }
                    float2 mm_270 = make_float2(m_269, m_269);
                    float2 pair_271 = make_float2(__uint_as_float(v4_268[0] << 16), __uint_as_float(v4_268[0] & 4294901760u));
                    float2 g_272;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_272) : "l"(*(const unsigned long long*)&pair_271), "l"(*(const unsigned long long*)&mm_270));
                    uint32_t _fp4_pair_39;
                    uint16_t _e2m1_pair_h_81;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(g_272.x), "f"(g_272.y));
                    _fp4_pair_39 = (uint32_t)_e2m1_pair_h_81;
                    float2 pair_273 = make_float2(__uint_as_float(v4_268[1] << 16), __uint_as_float(v4_268[1] & 4294901760u));
                    float2 g_274;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_274) : "l"(*(const unsigned long long*)&pair_273), "l"(*(const unsigned long long*)&mm_270));
                    uint32_t _fp4_pair_40;
                    uint16_t _e2m1_pair_h_82;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_82) : "f"(g_274.x), "f"(g_274.y));
                    _fp4_pair_40 = (uint32_t)_e2m1_pair_h_82;
                    float2 pair_275 = make_float2(__uint_as_float(v4_268[2] << 16), __uint_as_float(v4_268[2] & 4294901760u));
                    float2 g_276;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_276) : "l"(*(const unsigned long long*)&pair_275), "l"(*(const unsigned long long*)&mm_270));
                    uint32_t _fp4_pair_41;
                    uint16_t _e2m1_pair_h_83;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_83) : "f"(g_276.x), "f"(g_276.y));
                    _fp4_pair_41 = (uint32_t)_e2m1_pair_h_83;
                    float2 pair_277 = make_float2(__uint_as_float(v4_268[3] << 16), __uint_as_float(v4_268[3] & 4294901760u));
                    float2 g_278;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_278) : "l"(*(const unsigned long long*)&pair_277), "l"(*(const unsigned long long*)&mm_270));
                    uint32_t _fp4_e2m1_pack_pair_84;
                    uint16_t _e2m1_pair_h_85;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_85) : "f"(g_278.x), "f"(g_278.y));
                    _fp4_e2m1_pack_pair_84 = (uint32_t)_e2m1_pair_h_85;
                    unsigned int word_279 = _fp4_pair_39 | _fp4_pair_40 << 8 | _fp4_pair_41 << 16 | _fp4_e2m1_pack_pair_84 << 24;
                    unsigned int carry_280 = (word_279 & 2004318071) + 2004318071;
                    w_252[1] = word_279 & (carry_280 | 2004318071);
                    int cf_281 = f ^ 2;
                    unsigned int v4_282[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_282[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_282[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_282[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_282[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_247 + 16 * cf_281)));
                    float m_283 = m0_250;
                    if ((cf_281 & 2) != 0) {
                        m_283 = m1_251;
                    }
                    float2 mm_284 = make_float2(m_283, m_283);
                    float2 pair_285 = make_float2(__uint_as_float(v4_282[0] << 16), __uint_as_float(v4_282[0] & 4294901760u));
                    float2 g_286;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_286) : "l"(*(const unsigned long long*)&pair_285), "l"(*(const unsigned long long*)&mm_284));
                    uint32_t _fp4_pair_42;
                    uint16_t _e2m1_pair_h_86;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(g_286.x), "f"(g_286.y));
                    _fp4_pair_42 = (uint32_t)_e2m1_pair_h_86;
                    float2 pair_287 = make_float2(__uint_as_float(v4_282[1] << 16), __uint_as_float(v4_282[1] & 4294901760u));
                    float2 g_288;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_288) : "l"(*(const unsigned long long*)&pair_287), "l"(*(const unsigned long long*)&mm_284));
                    uint32_t _fp4_pair_43;
                    uint16_t _e2m1_pair_h_87;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_87) : "f"(g_288.x), "f"(g_288.y));
                    _fp4_pair_43 = (uint32_t)_e2m1_pair_h_87;
                    float2 pair_289 = make_float2(__uint_as_float(v4_282[2] << 16), __uint_as_float(v4_282[2] & 4294901760u));
                    float2 g_290;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_290) : "l"(*(const unsigned long long*)&pair_289), "l"(*(const unsigned long long*)&mm_284));
                    uint32_t _fp4_pair_44;
                    uint16_t _e2m1_pair_h_88;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_88) : "f"(g_290.x), "f"(g_290.y));
                    _fp4_pair_44 = (uint32_t)_e2m1_pair_h_88;
                    float2 pair_291 = make_float2(__uint_as_float(v4_282[3] << 16), __uint_as_float(v4_282[3] & 4294901760u));
                    float2 g_292;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_292) : "l"(*(const unsigned long long*)&pair_291), "l"(*(const unsigned long long*)&mm_284));
                    uint32_t _fp4_e2m1_pack_pair_89;
                    uint16_t _e2m1_pair_h_90;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(g_292.x), "f"(g_292.y));
                    _fp4_e2m1_pack_pair_89 = (uint32_t)_e2m1_pair_h_90;
                    unsigned int word_293 = _fp4_pair_42 | _fp4_pair_43 << 8 | _fp4_pair_44 << 16 | _fp4_e2m1_pack_pair_89 << 24;
                    unsigned int carry_294 = (word_293 & 2004318071) + 2004318071;
                    w_252[2] = word_293 & (carry_294 | 2004318071);
                    int cf_295 = f ^ 3;
                    unsigned int v4_296[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_296[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_296[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_296[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_296[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_247 + 16 * cf_295)));
                    float m_297 = m0_250;
                    if ((cf_295 & 2) != 0) {
                        m_297 = m1_251;
                    }
                    float2 mm_298 = make_float2(m_297, m_297);
                    float2 pair_299 = make_float2(__uint_as_float(v4_296[0] << 16), __uint_as_float(v4_296[0] & 4294901760u));
                    float2 g_300;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_300) : "l"(*(const unsigned long long*)&pair_299), "l"(*(const unsigned long long*)&mm_298));
                    uint32_t _fp4_pair_45;
                    uint16_t _e2m1_pair_h_91;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_91) : "f"(g_300.x), "f"(g_300.y));
                    _fp4_pair_45 = (uint32_t)_e2m1_pair_h_91;
                    float2 pair_301 = make_float2(__uint_as_float(v4_296[1] << 16), __uint_as_float(v4_296[1] & 4294901760u));
                    float2 g_302;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_302) : "l"(*(const unsigned long long*)&pair_301), "l"(*(const unsigned long long*)&mm_298));
                    uint32_t _fp4_pair_46;
                    uint16_t _e2m1_pair_h_92;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_92) : "f"(g_302.x), "f"(g_302.y));
                    _fp4_pair_46 = (uint32_t)_e2m1_pair_h_92;
                    float2 pair_303 = make_float2(__uint_as_float(v4_296[2] << 16), __uint_as_float(v4_296[2] & 4294901760u));
                    float2 g_304;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_304) : "l"(*(const unsigned long long*)&pair_303), "l"(*(const unsigned long long*)&mm_298));
                    uint32_t _fp4_pair_47;
                    uint16_t _e2m1_pair_h_93;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_93) : "f"(g_304.x), "f"(g_304.y));
                    _fp4_pair_47 = (uint32_t)_e2m1_pair_h_93;
                    float2 pair_305 = make_float2(__uint_as_float(v4_296[3] << 16), __uint_as_float(v4_296[3] & 4294901760u));
                    float2 g_306;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_306) : "l"(*(const unsigned long long*)&pair_305), "l"(*(const unsigned long long*)&mm_298));
                    uint32_t _fp4_e2m1_pack_pair_94;
                    uint16_t _e2m1_pair_h_95;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_95) : "f"(g_306.x), "f"(g_306.y));
                    _fp4_e2m1_pack_pair_94 = (uint32_t)_e2m1_pair_h_95;
                    unsigned int word_307 = _fp4_pair_45 | _fp4_pair_46 << 8 | _fp4_pair_47 << 16 | _fp4_e2m1_pack_pair_94 << 24;
                    unsigned int carry_308 = (word_307 & 2004318071) + 2004318071;
                    w_252[3] = word_307 & (carry_308 | 2004318071);
                    bool fodd_309 = (f & 1) != 0;
                    bool fhi_310 = (f & 2) != 0;
                    unsigned int t0_311 = w_252[0];
                    unsigned int t1_312 = w_252[1];
                    unsigned int t2_313 = w_252[2];
                    unsigned int t3_314 = w_252[3];
                    if (fodd_309) {
                        t0_311 = w_252[1];
                        t1_312 = w_252[0];
                        t2_313 = w_252[3];
                        t3_314 = w_252[2];
                    }
                    unsigned int q0_315 = t0_311;
                    unsigned int q1_316 = t1_312;
                    unsigned int q2_317 = t2_313;
                    unsigned int q3_318 = t3_314;
                    if (fhi_310) {
                        q0_315 = t2_313;
                        q1_316 = t3_314;
                        q2_317 = t0_311;
                        q3_318 = t1_312;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_246)), "r"(q0_315), "r"(q1_316), "r"(q2_317), "r"(q3_318) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_246)), "h"((uint16_t)(_e4m3x2_f32_3)) : "memory");
                    int pp2_319 = lane_2 + 128;
                    int base2_320 = xb_off + 64 * pp2_319;
                    float bm0_321 = __uint_as_float(bmp[4] << 16);
                    float bm1_322 = __uint_as_float(bmp[4] & 4294901760u);
                    float _fdiv_96 = __fdiv_rn(mul_f32_rn_noftz(bm1_322, inv), 6.0f);
                    float _fdiv_97 = __fdiv_rn(mul_f32_rn_noftz(bm0_321, inv), 6.0f);
                    uint16_t _e4m3x2_f32_4;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_4) : "f"(_fdiv_96), "f"(_fdiv_97));
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(_e4m3x2_f32_4)));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _fdiv_rn_9 = __fdiv_rn(inv, _cvt_f32_f16_8);
                    float _fdiv_rn_10 = __fdiv_rn(inv, _cvt_f32_f16_9);
                    float m0_323 = 0.0f;
                    float m1_324 = 0.0f;
                    if (_cvt_f32_f16_8 > 0.0f) {
                        m0_323 = _fdiv_rn_9;
                    }
                    if (_cvt_f32_f16_9 > 0.0f) {
                        m1_324 = _fdiv_rn_10;
                    }
                    unsigned int w_325[4];
                    int cf_326 = f ^ 0;
                    unsigned int v4_327[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_327[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_327[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_327[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_327[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_320 + 16 * cf_326)));
                    float m_328 = m0_323;
                    if ((cf_326 & 2) != 0) {
                        m_328 = m1_324;
                    }
                    float2 mm_329 = make_float2(m_328, m_328);
                    float2 pair_330 = make_float2(__uint_as_float(v4_327[0] << 16), __uint_as_float(v4_327[0] & 4294901760u));
                    float2 g_331;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_331) : "l"(*(const unsigned long long*)&pair_330), "l"(*(const unsigned long long*)&mm_329));
                    uint32_t _fp4_pair_48;
                    uint16_t _e2m1_pair_h_98;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_98) : "f"(g_331.x), "f"(g_331.y));
                    _fp4_pair_48 = (uint32_t)_e2m1_pair_h_98;
                    float2 pair_332 = make_float2(__uint_as_float(v4_327[1] << 16), __uint_as_float(v4_327[1] & 4294901760u));
                    float2 g_333;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_333) : "l"(*(const unsigned long long*)&pair_332), "l"(*(const unsigned long long*)&mm_329));
                    uint32_t _fp4_pair_49;
                    uint16_t _e2m1_pair_h_99;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_99) : "f"(g_333.x), "f"(g_333.y));
                    _fp4_pair_49 = (uint32_t)_e2m1_pair_h_99;
                    float2 pair_334 = make_float2(__uint_as_float(v4_327[2] << 16), __uint_as_float(v4_327[2] & 4294901760u));
                    float2 g_335;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_335) : "l"(*(const unsigned long long*)&pair_334), "l"(*(const unsigned long long*)&mm_329));
                    uint32_t _fp4_pair_50;
                    uint16_t _e2m1_pair_h_100;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_100) : "f"(g_335.x), "f"(g_335.y));
                    _fp4_pair_50 = (uint32_t)_e2m1_pair_h_100;
                    float2 pair_336 = make_float2(__uint_as_float(v4_327[3] << 16), __uint_as_float(v4_327[3] & 4294901760u));
                    float2 g_337;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_337) : "l"(*(const unsigned long long*)&pair_336), "l"(*(const unsigned long long*)&mm_329));
                    uint32_t _fp4_e2m1_pack_pair_101;
                    uint16_t _e2m1_pair_h_102;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(g_337.x), "f"(g_337.y));
                    _fp4_e2m1_pack_pair_101 = (uint32_t)_e2m1_pair_h_102;
                    unsigned int word_338 = _fp4_pair_48 | _fp4_pair_49 << 8 | _fp4_pair_50 << 16 | _fp4_e2m1_pack_pair_101 << 24;
                    unsigned int carry_339 = (word_338 & 2004318071) + 2004318071;
                    w_325[0] = word_338 & (carry_339 | 2004318071);
                    int cf_340 = f ^ 1;
                    unsigned int v4_341[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_341[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_341[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_341[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_341[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_320 + 16 * cf_340)));
                    float m_342 = m0_323;
                    if ((cf_340 & 2) != 0) {
                        m_342 = m1_324;
                    }
                    float2 mm_343 = make_float2(m_342, m_342);
                    float2 pair_344 = make_float2(__uint_as_float(v4_341[0] << 16), __uint_as_float(v4_341[0] & 4294901760u));
                    float2 g_345;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_345) : "l"(*(const unsigned long long*)&pair_344), "l"(*(const unsigned long long*)&mm_343));
                    uint32_t _fp4_pair_51;
                    uint16_t _e2m1_pair_h_103;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_103) : "f"(g_345.x), "f"(g_345.y));
                    _fp4_pair_51 = (uint32_t)_e2m1_pair_h_103;
                    float2 pair_346 = make_float2(__uint_as_float(v4_341[1] << 16), __uint_as_float(v4_341[1] & 4294901760u));
                    float2 g_347;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_347) : "l"(*(const unsigned long long*)&pair_346), "l"(*(const unsigned long long*)&mm_343));
                    uint32_t _fp4_pair_52;
                    uint16_t _e2m1_pair_h_104;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_104) : "f"(g_347.x), "f"(g_347.y));
                    _fp4_pair_52 = (uint32_t)_e2m1_pair_h_104;
                    float2 pair_348 = make_float2(__uint_as_float(v4_341[2] << 16), __uint_as_float(v4_341[2] & 4294901760u));
                    float2 g_349;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_349) : "l"(*(const unsigned long long*)&pair_348), "l"(*(const unsigned long long*)&mm_343));
                    uint32_t _fp4_pair_53;
                    uint16_t _e2m1_pair_h_105;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_105) : "f"(g_349.x), "f"(g_349.y));
                    _fp4_pair_53 = (uint32_t)_e2m1_pair_h_105;
                    float2 pair_350 = make_float2(__uint_as_float(v4_341[3] << 16), __uint_as_float(v4_341[3] & 4294901760u));
                    float2 g_351;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_351) : "l"(*(const unsigned long long*)&pair_350), "l"(*(const unsigned long long*)&mm_343));
                    uint32_t _fp4_e2m1_pack_pair_106;
                    uint16_t _e2m1_pair_h_107;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_107) : "f"(g_351.x), "f"(g_351.y));
                    _fp4_e2m1_pack_pair_106 = (uint32_t)_e2m1_pair_h_107;
                    unsigned int word_352 = _fp4_pair_51 | _fp4_pair_52 << 8 | _fp4_pair_53 << 16 | _fp4_e2m1_pack_pair_106 << 24;
                    unsigned int carry_353 = (word_352 & 2004318071) + 2004318071;
                    w_325[1] = word_352 & (carry_353 | 2004318071);
                    int cf_354 = f ^ 2;
                    unsigned int v4_355[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_355[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_355[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_355[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_355[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_320 + 16 * cf_354)));
                    float m_356 = m0_323;
                    if ((cf_354 & 2) != 0) {
                        m_356 = m1_324;
                    }
                    float2 mm_357 = make_float2(m_356, m_356);
                    float2 pair_358 = make_float2(__uint_as_float(v4_355[0] << 16), __uint_as_float(v4_355[0] & 4294901760u));
                    float2 g_359;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_359) : "l"(*(const unsigned long long*)&pair_358), "l"(*(const unsigned long long*)&mm_357));
                    uint32_t _fp4_pair_54;
                    uint16_t _e2m1_pair_h_108;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(g_359.x), "f"(g_359.y));
                    _fp4_pair_54 = (uint32_t)_e2m1_pair_h_108;
                    float2 pair_360 = make_float2(__uint_as_float(v4_355[1] << 16), __uint_as_float(v4_355[1] & 4294901760u));
                    float2 g_361;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_361) : "l"(*(const unsigned long long*)&pair_360), "l"(*(const unsigned long long*)&mm_357));
                    uint32_t _fp4_pair_55;
                    uint16_t _e2m1_pair_h_109;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_109) : "f"(g_361.x), "f"(g_361.y));
                    _fp4_pair_55 = (uint32_t)_e2m1_pair_h_109;
                    float2 pair_362 = make_float2(__uint_as_float(v4_355[2] << 16), __uint_as_float(v4_355[2] & 4294901760u));
                    float2 g_363;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_363) : "l"(*(const unsigned long long*)&pair_362), "l"(*(const unsigned long long*)&mm_357));
                    uint32_t _fp4_pair_56;
                    uint16_t _e2m1_pair_h_110;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_110) : "f"(g_363.x), "f"(g_363.y));
                    _fp4_pair_56 = (uint32_t)_e2m1_pair_h_110;
                    float2 pair_364 = make_float2(__uint_as_float(v4_355[3] << 16), __uint_as_float(v4_355[3] & 4294901760u));
                    float2 g_365;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_365) : "l"(*(const unsigned long long*)&pair_364), "l"(*(const unsigned long long*)&mm_357));
                    uint32_t _fp4_e2m1_pack_pair_111;
                    uint16_t _e2m1_pair_h_112;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_112) : "f"(g_365.x), "f"(g_365.y));
                    _fp4_e2m1_pack_pair_111 = (uint32_t)_e2m1_pair_h_112;
                    unsigned int word_366 = _fp4_pair_54 | _fp4_pair_55 << 8 | _fp4_pair_56 << 16 | _fp4_e2m1_pack_pair_111 << 24;
                    unsigned int carry_367 = (word_366 & 2004318071) + 2004318071;
                    w_325[2] = word_366 & (carry_367 | 2004318071);
                    int cf_368 = f ^ 3;
                    unsigned int v4_369[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_369[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_369[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_369[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_369[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_320 + 16 * cf_368)));
                    float m_370 = m0_323;
                    if ((cf_368 & 2) != 0) {
                        m_370 = m1_324;
                    }
                    float2 mm_371 = make_float2(m_370, m_370);
                    float2 pair_372 = make_float2(__uint_as_float(v4_369[0] << 16), __uint_as_float(v4_369[0] & 4294901760u));
                    float2 g_373;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_373) : "l"(*(const unsigned long long*)&pair_372), "l"(*(const unsigned long long*)&mm_371));
                    uint32_t _fp4_pair_57;
                    uint16_t _e2m1_pair_h_113;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(g_373.x), "f"(g_373.y));
                    _fp4_pair_57 = (uint32_t)_e2m1_pair_h_113;
                    float2 pair_374 = make_float2(__uint_as_float(v4_369[1] << 16), __uint_as_float(v4_369[1] & 4294901760u));
                    float2 g_375;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_375) : "l"(*(const unsigned long long*)&pair_374), "l"(*(const unsigned long long*)&mm_371));
                    uint32_t _fp4_pair_58;
                    uint16_t _e2m1_pair_h_114;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_114) : "f"(g_375.x), "f"(g_375.y));
                    _fp4_pair_58 = (uint32_t)_e2m1_pair_h_114;
                    float2 pair_376 = make_float2(__uint_as_float(v4_369[2] << 16), __uint_as_float(v4_369[2] & 4294901760u));
                    float2 g_377;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_377) : "l"(*(const unsigned long long*)&pair_376), "l"(*(const unsigned long long*)&mm_371));
                    uint32_t _fp4_pair_59;
                    uint16_t _e2m1_pair_h_115;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_115) : "f"(g_377.x), "f"(g_377.y));
                    _fp4_pair_59 = (uint32_t)_e2m1_pair_h_115;
                    float2 pair_378 = make_float2(__uint_as_float(v4_369[3] << 16), __uint_as_float(v4_369[3] & 4294901760u));
                    float2 g_379;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_379) : "l"(*(const unsigned long long*)&pair_378), "l"(*(const unsigned long long*)&mm_371));
                    uint32_t _fp4_e2m1_pack_pair_116;
                    uint16_t _e2m1_pair_h_117;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_117) : "f"(g_379.x), "f"(g_379.y));
                    _fp4_e2m1_pack_pair_116 = (uint32_t)_e2m1_pair_h_117;
                    unsigned int word_380 = _fp4_pair_57 | _fp4_pair_58 << 8 | _fp4_pair_59 << 16 | _fp4_e2m1_pack_pair_116 << 24;
                    unsigned int carry_381 = (word_380 & 2004318071) + 2004318071;
                    w_325[3] = word_380 & (carry_381 | 2004318071);
                    bool fodd_382 = (f & 1) != 0;
                    bool fhi_383 = (f & 2) != 0;
                    unsigned int t0_384 = w_325[0];
                    unsigned int t1_385 = w_325[1];
                    unsigned int t2_386 = w_325[2];
                    unsigned int t3_387 = w_325[3];
                    if (fodd_382) {
                        t0_384 = w_325[1];
                        t1_385 = w_325[0];
                        t2_386 = w_325[3];
                        t3_387 = w_325[2];
                    }
                    unsigned int q0_388 = t0_384;
                    unsigned int q1_389 = t1_385;
                    unsigned int q2_390 = t2_386;
                    unsigned int q3_391 = t3_387;
                    if (fhi_383) {
                        q0_388 = t2_386;
                        q1_389 = t3_387;
                        q2_390 = t0_384;
                        q3_391 = t1_385;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_319)), "r"(q0_388), "r"(q1_389), "r"(q2_390), "r"(q3_391) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_319)), "h"((uint16_t)(_e4m3x2_f32_4)) : "memory");
                    int pp2_392 = lane_2 + 160;
                    int base2_393 = xb_off + 64 * pp2_392;
                    float bm0_394 = __uint_as_float(bmp[5] << 16);
                    float bm1_395 = __uint_as_float(bmp[5] & 4294901760u);
                    float _fdiv_118 = __fdiv_rn(mul_f32_rn_noftz(bm1_395, inv), 6.0f);
                    float _fdiv_119 = __fdiv_rn(mul_f32_rn_noftz(bm0_394, inv), 6.0f);
                    uint16_t _e4m3x2_f32_5;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_5) : "f"(_fdiv_118), "f"(_fdiv_119));
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(_e4m3x2_f32_5)));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _fdiv_rn_11 = __fdiv_rn(inv, _cvt_f32_f16_10);
                    float _fdiv_rn_12 = __fdiv_rn(inv, _cvt_f32_f16_11);
                    float m0_396 = 0.0f;
                    float m1_397 = 0.0f;
                    if (_cvt_f32_f16_10 > 0.0f) {
                        m0_396 = _fdiv_rn_11;
                    }
                    if (_cvt_f32_f16_11 > 0.0f) {
                        m1_397 = _fdiv_rn_12;
                    }
                    unsigned int w_398[4];
                    int cf_399 = f ^ 0;
                    unsigned int v4_400[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_400[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_400[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_400[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_400[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_393 + 16 * cf_399)));
                    float m_401 = m0_396;
                    if ((cf_399 & 2) != 0) {
                        m_401 = m1_397;
                    }
                    float2 mm_402 = make_float2(m_401, m_401);
                    float2 pair_403 = make_float2(__uint_as_float(v4_400[0] << 16), __uint_as_float(v4_400[0] & 4294901760u));
                    float2 g_404;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_404) : "l"(*(const unsigned long long*)&pair_403), "l"(*(const unsigned long long*)&mm_402));
                    uint32_t _fp4_pair_60;
                    uint16_t _e2m1_pair_h_120;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_120) : "f"(g_404.x), "f"(g_404.y));
                    _fp4_pair_60 = (uint32_t)_e2m1_pair_h_120;
                    float2 pair_405 = make_float2(__uint_as_float(v4_400[1] << 16), __uint_as_float(v4_400[1] & 4294901760u));
                    float2 g_406;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_406) : "l"(*(const unsigned long long*)&pair_405), "l"(*(const unsigned long long*)&mm_402));
                    uint32_t _fp4_pair_61;
                    uint16_t _e2m1_pair_h_121;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_121) : "f"(g_406.x), "f"(g_406.y));
                    _fp4_pair_61 = (uint32_t)_e2m1_pair_h_121;
                    float2 pair_407 = make_float2(__uint_as_float(v4_400[2] << 16), __uint_as_float(v4_400[2] & 4294901760u));
                    float2 g_408;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_408) : "l"(*(const unsigned long long*)&pair_407), "l"(*(const unsigned long long*)&mm_402));
                    uint32_t _fp4_pair_62;
                    uint16_t _e2m1_pair_h_122;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_122) : "f"(g_408.x), "f"(g_408.y));
                    _fp4_pair_62 = (uint32_t)_e2m1_pair_h_122;
                    float2 pair_409 = make_float2(__uint_as_float(v4_400[3] << 16), __uint_as_float(v4_400[3] & 4294901760u));
                    float2 g_410;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_410) : "l"(*(const unsigned long long*)&pair_409), "l"(*(const unsigned long long*)&mm_402));
                    uint32_t _fp4_e2m1_pack_pair_123;
                    uint16_t _e2m1_pair_h_124;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_124) : "f"(g_410.x), "f"(g_410.y));
                    _fp4_e2m1_pack_pair_123 = (uint32_t)_e2m1_pair_h_124;
                    unsigned int word_411 = _fp4_pair_60 | _fp4_pair_61 << 8 | _fp4_pair_62 << 16 | _fp4_e2m1_pack_pair_123 << 24;
                    unsigned int carry_412 = (word_411 & 2004318071) + 2004318071;
                    w_398[0] = word_411 & (carry_412 | 2004318071);
                    int cf_413 = f ^ 1;
                    unsigned int v4_414[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_414[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_414[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_414[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_414[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_393 + 16 * cf_413)));
                    float m_415 = m0_396;
                    if ((cf_413 & 2) != 0) {
                        m_415 = m1_397;
                    }
                    float2 mm_416 = make_float2(m_415, m_415);
                    float2 pair_417 = make_float2(__uint_as_float(v4_414[0] << 16), __uint_as_float(v4_414[0] & 4294901760u));
                    float2 g_418;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_418) : "l"(*(const unsigned long long*)&pair_417), "l"(*(const unsigned long long*)&mm_416));
                    uint32_t _fp4_pair_63;
                    uint16_t _e2m1_pair_h_125;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_125) : "f"(g_418.x), "f"(g_418.y));
                    _fp4_pair_63 = (uint32_t)_e2m1_pair_h_125;
                    float2 pair_419 = make_float2(__uint_as_float(v4_414[1] << 16), __uint_as_float(v4_414[1] & 4294901760u));
                    float2 g_420;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_420) : "l"(*(const unsigned long long*)&pair_419), "l"(*(const unsigned long long*)&mm_416));
                    uint32_t _fp4_pair_64;
                    uint16_t _e2m1_pair_h_126;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_126) : "f"(g_420.x), "f"(g_420.y));
                    _fp4_pair_64 = (uint32_t)_e2m1_pair_h_126;
                    float2 pair_421 = make_float2(__uint_as_float(v4_414[2] << 16), __uint_as_float(v4_414[2] & 4294901760u));
                    float2 g_422;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_422) : "l"(*(const unsigned long long*)&pair_421), "l"(*(const unsigned long long*)&mm_416));
                    uint32_t _fp4_pair_65;
                    uint16_t _e2m1_pair_h_127;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_127) : "f"(g_422.x), "f"(g_422.y));
                    _fp4_pair_65 = (uint32_t)_e2m1_pair_h_127;
                    float2 pair_423 = make_float2(__uint_as_float(v4_414[3] << 16), __uint_as_float(v4_414[3] & 4294901760u));
                    float2 g_424;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_424) : "l"(*(const unsigned long long*)&pair_423), "l"(*(const unsigned long long*)&mm_416));
                    uint32_t _fp4_e2m1_pack_pair_128;
                    uint16_t _e2m1_pair_h_129;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_129) : "f"(g_424.x), "f"(g_424.y));
                    _fp4_e2m1_pack_pair_128 = (uint32_t)_e2m1_pair_h_129;
                    unsigned int word_425 = _fp4_pair_63 | _fp4_pair_64 << 8 | _fp4_pair_65 << 16 | _fp4_e2m1_pack_pair_128 << 24;
                    unsigned int carry_426 = (word_425 & 2004318071) + 2004318071;
                    w_398[1] = word_425 & (carry_426 | 2004318071);
                    int cf_427 = f ^ 2;
                    unsigned int v4_428[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_428[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_428[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_428[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_428[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_393 + 16 * cf_427)));
                    float m_429 = m0_396;
                    if ((cf_427 & 2) != 0) {
                        m_429 = m1_397;
                    }
                    float2 mm_430 = make_float2(m_429, m_429);
                    float2 pair_431 = make_float2(__uint_as_float(v4_428[0] << 16), __uint_as_float(v4_428[0] & 4294901760u));
                    float2 g_432;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_432) : "l"(*(const unsigned long long*)&pair_431), "l"(*(const unsigned long long*)&mm_430));
                    uint32_t _fp4_pair_66;
                    uint16_t _e2m1_pair_h_130;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_130) : "f"(g_432.x), "f"(g_432.y));
                    _fp4_pair_66 = (uint32_t)_e2m1_pair_h_130;
                    float2 pair_433 = make_float2(__uint_as_float(v4_428[1] << 16), __uint_as_float(v4_428[1] & 4294901760u));
                    float2 g_434;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_434) : "l"(*(const unsigned long long*)&pair_433), "l"(*(const unsigned long long*)&mm_430));
                    uint32_t _fp4_pair_67;
                    uint16_t _e2m1_pair_h_131;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_131) : "f"(g_434.x), "f"(g_434.y));
                    _fp4_pair_67 = (uint32_t)_e2m1_pair_h_131;
                    float2 pair_435 = make_float2(__uint_as_float(v4_428[2] << 16), __uint_as_float(v4_428[2] & 4294901760u));
                    float2 g_436;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_436) : "l"(*(const unsigned long long*)&pair_435), "l"(*(const unsigned long long*)&mm_430));
                    uint32_t _fp4_pair_68;
                    uint16_t _e2m1_pair_h_132;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_132) : "f"(g_436.x), "f"(g_436.y));
                    _fp4_pair_68 = (uint32_t)_e2m1_pair_h_132;
                    float2 pair_437 = make_float2(__uint_as_float(v4_428[3] << 16), __uint_as_float(v4_428[3] & 4294901760u));
                    float2 g_438;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_438) : "l"(*(const unsigned long long*)&pair_437), "l"(*(const unsigned long long*)&mm_430));
                    uint32_t _fp4_e2m1_pack_pair_133;
                    uint16_t _e2m1_pair_h_134;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_134) : "f"(g_438.x), "f"(g_438.y));
                    _fp4_e2m1_pack_pair_133 = (uint32_t)_e2m1_pair_h_134;
                    unsigned int word_439 = _fp4_pair_66 | _fp4_pair_67 << 8 | _fp4_pair_68 << 16 | _fp4_e2m1_pack_pair_133 << 24;
                    unsigned int carry_440 = (word_439 & 2004318071) + 2004318071;
                    w_398[2] = word_439 & (carry_440 | 2004318071);
                    int cf_441 = f ^ 3;
                    unsigned int v4_442[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_442[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_442[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_442[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_442[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_393 + 16 * cf_441)));
                    float m_443 = m0_396;
                    if ((cf_441 & 2) != 0) {
                        m_443 = m1_397;
                    }
                    float2 mm_444 = make_float2(m_443, m_443);
                    float2 pair_445 = make_float2(__uint_as_float(v4_442[0] << 16), __uint_as_float(v4_442[0] & 4294901760u));
                    float2 g_446;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_446) : "l"(*(const unsigned long long*)&pair_445), "l"(*(const unsigned long long*)&mm_444));
                    uint32_t _fp4_pair_69;
                    uint16_t _e2m1_pair_h_135;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_135) : "f"(g_446.x), "f"(g_446.y));
                    _fp4_pair_69 = (uint32_t)_e2m1_pair_h_135;
                    float2 pair_447 = make_float2(__uint_as_float(v4_442[1] << 16), __uint_as_float(v4_442[1] & 4294901760u));
                    float2 g_448;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_448) : "l"(*(const unsigned long long*)&pair_447), "l"(*(const unsigned long long*)&mm_444));
                    uint32_t _fp4_pair_70;
                    uint16_t _e2m1_pair_h_136;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_136) : "f"(g_448.x), "f"(g_448.y));
                    _fp4_pair_70 = (uint32_t)_e2m1_pair_h_136;
                    float2 pair_449 = make_float2(__uint_as_float(v4_442[2] << 16), __uint_as_float(v4_442[2] & 4294901760u));
                    float2 g_450;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_450) : "l"(*(const unsigned long long*)&pair_449), "l"(*(const unsigned long long*)&mm_444));
                    uint32_t _fp4_pair_71;
                    uint16_t _e2m1_pair_h_137;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_137) : "f"(g_450.x), "f"(g_450.y));
                    _fp4_pair_71 = (uint32_t)_e2m1_pair_h_137;
                    float2 pair_451 = make_float2(__uint_as_float(v4_442[3] << 16), __uint_as_float(v4_442[3] & 4294901760u));
                    float2 g_452;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_452) : "l"(*(const unsigned long long*)&pair_451), "l"(*(const unsigned long long*)&mm_444));
                    uint32_t _fp4_e2m1_pack_pair_138;
                    uint16_t _e2m1_pair_h_139;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_139) : "f"(g_452.x), "f"(g_452.y));
                    _fp4_e2m1_pack_pair_138 = (uint32_t)_e2m1_pair_h_139;
                    unsigned int word_453 = _fp4_pair_69 | _fp4_pair_70 << 8 | _fp4_pair_71 << 16 | _fp4_e2m1_pack_pair_138 << 24;
                    unsigned int carry_454 = (word_453 & 2004318071) + 2004318071;
                    w_398[3] = word_453 & (carry_454 | 2004318071);
                    bool fodd_455 = (f & 1) != 0;
                    bool fhi_456 = (f & 2) != 0;
                    unsigned int t0_457 = w_398[0];
                    unsigned int t1_458 = w_398[1];
                    unsigned int t2_459 = w_398[2];
                    unsigned int t3_460 = w_398[3];
                    if (fodd_455) {
                        t0_457 = w_398[1];
                        t1_458 = w_398[0];
                        t2_459 = w_398[3];
                        t3_460 = w_398[2];
                    }
                    unsigned int q0_461 = t0_457;
                    unsigned int q1_462 = t1_458;
                    unsigned int q2_463 = t2_459;
                    unsigned int q3_464 = t3_460;
                    if (fhi_456) {
                        q0_461 = t2_459;
                        q1_462 = t3_460;
                        q2_463 = t0_457;
                        q3_464 = t1_458;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_392)), "r"(q0_461), "r"(q1_462), "r"(q2_463), "r"(q3_464) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_392)), "h"((uint16_t)(_e4m3x2_f32_5)) : "memory");
                    rs = rs_50;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 < 4) {
                        {
                            uint32_t _addr_140 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_2));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_140), "f"(rs) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_141 = reinterpret_cast<void*>(peer_wire + (v_base + (rank64 * cap64 + row64) * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_141), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_142 = reinterpret_cast<void*>(peer_wire + (s_base + (rank64 * cap64 + row64) * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_142), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_143 = reinterpret_cast<void*>(peer_rs + (m_base + (rank64 * cap64 + row64) * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_143), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + warp_1 + 10 * kd);
                    if (lane_2 == 0) {
                        mbarrier_arrive_expect_tx(rb_full_addr + (warp_1) * 8, 12288);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)rb_off, res_in + (drow * 6144), 12288, rb_full_addr + (warp_1) * 8);
                    }
                    int dob_off = own_off + kd % 2 * 3472;
                    float own_rs = wsm32[dob_off + 3072 + 384 >> 2];
                    bool first = rank == 0;
                    long long src64 = peer64;
                    long long v_row = v_base + (src64 * cap64 + drow) * 3072;
                    long long s_row = s_base + (src64 * cap64 + drow) * 384;
                    long long m_row = m_base + (src64 * cap64 + drow) * 16;
                    unsigned int rsw[1];
                    uint32_t _sysv_u32_0;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = _sysv_u32_0;
                    float ssx = 0.0f;
                    float ssy = 0.0f;
                    #pragma unroll 1
                    for (int j0 = 0; j0 < 6; j0 += 2) {
                        int pp0 = lane_2 + 32 * j0;
                        int pp1 = pp0 + 32;
                        uint32_t _fp4_wire_values_0[8];
                        uint32_t _fp4_wire_scales_0[2];
                        uint16_t _wire_h16_144_0;
                        uint16_t _wire_h16_144_1;
                        uint32_t _wire_vpend_144 = 3u;
                        uint32_t _wire_spend_144 = 3u;
                        do {
                            if (_wire_vpend_144 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0)))) : "memory");
                            }
                            if (_wire_vpend_144 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1)))) : "memory");
                            }
                            if (_wire_spend_144 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_144_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0)))) : "memory");
                                _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_144_0;
                            }
                            if (_wire_spend_144 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_144_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1)))) : "memory");
                                _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_144_1;
                            }
                            _wire_vpend_144 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_144 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_144 | _wire_spend_144) != 0u);
                        if (j0 == 0) {
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
                        }
                        if (j0 == 0) {
                            mbarrier_wait(rb_full_addr + (warp_1) * 8, kd & 1);
                        }
                        float rs_peer_f = __uint_as_float(rsw[0]);
                        float rs_a_f = ((first) ? own_rs : rs_peer_f);
                        float rs_b_f = ((first) ? rs_peer_f : own_rs);
                        unsigned int own[4];
                        unsigned int own_sc[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp0)));
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp0)));
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
                        float acc[32] = {0};
                        bool fhi_1 = (f & 2) != 0;
                        bool fodd_1 = (f & 1) != 0;
                        bool fhi_0 = (f & 2) != 0;
                        unsigned int s[4];
                        s[0] = a_1[0];
                        s[1] = a_1[1];
                        s[2] = a_1[2];
                        s[3] = a_1[3];
                        if (fodd_1) {
                            s[0] = a_1[1];
                            s[1] = a_1[0];
                            s[2] = a_1[3];
                            s[3] = a_1[2];
                        }
                        unsigned int r[4];
                        r[0] = s[0];
                        r[1] = s[1];
                        r[2] = s[2];
                        r[3] = s[3];
                        if (fhi_0) {
                            r[0] = s[2];
                            r[1] = s[3];
                            r[2] = s[0];
                            r[3] = s[1];
                        }
                        bool fodd_1_1 = (f & 1) != 0;
                        bool fhi_2 = (f & 2) != 0;
                        unsigned int s_3[4];
                        s_3[0] = b_1[0];
                        s_3[1] = b_1[1];
                        s_3[2] = b_1[2];
                        s_3[3] = b_1[3];
                        if (fodd_1_1) {
                            s_3[0] = b_1[1];
                            s_3[1] = b_1[0];
                            s_3[2] = b_1[3];
                            s_3[3] = b_1[2];
                        }
                        unsigned int r_4[4];
                        r_4[0] = s_3[0];
                        r_4[1] = s_3[1];
                        r_4[2] = s_3[2];
                        r_4[3] = s_3[3];
                        if (fhi_2) {
                            r_4[0] = s_3[2];
                            r_4[1] = s_3[3];
                            r_4[2] = s_3[0];
                            r_4[3] = s_3[1];
                        }
                        uint32_t _e4m3x2_to_f16x2_6;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(a_sc)));
                        float _cvt_f32_f16_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                        float _cvt_f32_f16_13;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_12), "f"(rs_a_f));
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_13), "f"(rs_a_f));
                        float lo = _mul_0;
                        float hi = _mul_1;
                        if (fhi_1) {
                            lo = _mul_1;
                            hi = _mul_0;
                        }
                        float2 scale0 = make_float2(lo, lo);
                        float2 scale1 = make_float2(hi, hi);
                        float _unpack_fp4_0[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[0]));
                            (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(r[0]));
                            #endif
                        }
                        float2 a_5 = make_float2(acc[0], acc[1]);
                        float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                        float2 r_6 = fma_f32x2_rn_noftz(q, scale0, a_5);
                        acc[0] = r_6.x;
                        acc[1] = r_6.y;
                        float2 a_7 = make_float2(acc[2], acc[3]);
                        float2 q_8 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                        float2 r_9 = fma_f32x2_rn_noftz(q_8, scale0, a_7);
                        acc[2] = r_9.x;
                        acc[3] = r_9.y;
                        float2 a_10 = make_float2(acc[4], acc[5]);
                        float2 q_11 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                        float2 r_12 = fma_f32x2_rn_noftz(q_11, scale0, a_10);
                        acc[4] = r_12.x;
                        acc[5] = r_12.y;
                        float2 a_13 = make_float2(acc[6], acc[7]);
                        float2 q_14 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                        float2 r_15 = fma_f32x2_rn_noftz(q_14, scale0, a_13);
                        acc[6] = r_15.x;
                        acc[7] = r_15.y;
                        float _unpack_fp4_1[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[1 + 0]));
                            (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(r[1 + 0]));
                            #endif
                        }
                        float2 a_16 = make_float2(acc[8], acc[9]);
                        float2 q_17 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                        float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0, a_16);
                        acc[8] = r_18.x;
                        acc[9] = r_18.y;
                        float2 a_19 = make_float2(acc[10], acc[11]);
                        float2 q_20 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                        float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0, a_19);
                        acc[10] = r_21.x;
                        acc[11] = r_21.y;
                        float2 a_22_1 = make_float2(acc[12], acc[13]);
                        float2 q_23 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                        float2 r_24 = fma_f32x2_rn_noftz(q_23, scale0, a_22_1);
                        acc[12] = r_24.x;
                        acc[13] = r_24.y;
                        float2 a_25 = make_float2(acc[14], acc[15]);
                        float2 q_26 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                        float2 r_27 = fma_f32x2_rn_noftz(q_26, scale0, a_25);
                        acc[14] = r_27.x;
                        acc[15] = r_27.y;
                        float _unpack_fp4_2[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[2 + 0]));
                            (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(r[2 + 0]));
                            #endif
                        }
                        float2 a_28 = make_float2(acc[16], acc[17]);
                        float2 q_29 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                        float2 r_30 = fma_f32x2_rn_noftz(q_29, scale1, a_28);
                        acc[16] = r_30.x;
                        acc[17] = r_30.y;
                        float2 a_31 = make_float2(acc[18], acc[19]);
                        float2 q_32 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                        float2 r_33 = fma_f32x2_rn_noftz(q_32, scale1, a_31);
                        acc[18] = r_33.x;
                        acc[19] = r_33.y;
                        float2 a_34 = make_float2(acc[20], acc[21]);
                        float2 q_35 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                        float2 r_36 = fma_f32x2_rn_noftz(q_35, scale1, a_34);
                        acc[20] = r_36.x;
                        acc[21] = r_36.y;
                        float2 a_37 = make_float2(acc[22], acc[23]);
                        float2 q_38 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                        float2 r_39 = fma_f32x2_rn_noftz(q_38, scale1, a_37);
                        acc[22] = r_39.x;
                        acc[23] = r_39.y;
                        float _unpack_fp4_3[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[3 + 0]));
                            (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(r[3 + 0]));
                            #endif
                        }
                        float2 a_40 = make_float2(acc[24], acc[25]);
                        float2 q_41 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                        float2 r_42 = fma_f32x2_rn_noftz(q_41, scale1, a_40);
                        acc[24] = r_42.x;
                        acc[25] = r_42.y;
                        float2 a_43 = make_float2(acc[26], acc[27]);
                        float2 q_44 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                        float2 r_45 = fma_f32x2_rn_noftz(q_44, scale1, a_43);
                        acc[26] = r_45.x;
                        acc[27] = r_45.y;
                        float2 a_46 = make_float2(acc[28], acc[29]);
                        float2 q_47 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                        float2 r_48 = fma_f32x2_rn_noftz(q_47, scale1, a_46);
                        acc[28] = r_48.x;
                        acc[29] = r_48.y;
                        float2 a_49 = make_float2(acc[30], acc[31]);
                        float2 q_50 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                        float2 r_51 = fma_f32x2_rn_noftz(q_50, scale1, a_49);
                        acc[30] = r_51.x;
                        acc[31] = r_51.y;
                        uint32_t _e4m3x2_to_f16x2_7;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(b_sc)));
                        float _cvt_f32_f16_14;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                        float _cvt_f32_f16_15;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                        float _mul_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_14), "f"(rs_b_f));
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_15), "f"(rs_b_f));
                        float lo_52 = _mul_2;
                        float hi_53 = _mul_3;
                        if (fhi_1) {
                            lo_52 = _mul_3;
                            hi_53 = _mul_2;
                        }
                        float2 scale0_54 = make_float2(lo_52, lo_52);
                        float2 scale1_55 = make_float2(hi_53, hi_53);
                        float _unpack_fp4_4[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[0]));
                            (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(r_4[0]));
                            #endif
                        }
                        float2 a_56 = make_float2(acc[0], acc[1]);
                        float2 q_57 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                        float2 r_58 = fma_f32x2_rn_noftz(q_57, scale0_54, a_56);
                        acc[0] = r_58.x;
                        acc[1] = r_58.y;
                        float2 a_59 = make_float2(acc[2], acc[3]);
                        float2 q_60 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                        float2 r_61 = fma_f32x2_rn_noftz(q_60, scale0_54, a_59);
                        acc[2] = r_61.x;
                        acc[3] = r_61.y;
                        float2 a_62 = make_float2(acc[4], acc[5]);
                        float2 q_63 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                        float2 r_64 = fma_f32x2_rn_noftz(q_63, scale0_54, a_62);
                        acc[4] = r_64.x;
                        acc[5] = r_64.y;
                        float2 a_65 = make_float2(acc[6], acc[7]);
                        float2 q_66 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                        float2 r_67 = fma_f32x2_rn_noftz(q_66, scale0_54, a_65);
                        acc[6] = r_67.x;
                        acc[7] = r_67.y;
                        float _unpack_fp4_5[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[1 + 0]));
                            (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(r_4[1 + 0]));
                            #endif
                        }
                        float2 a_68 = make_float2(acc[8], acc[9]);
                        float2 q_69 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                        float2 r_70 = fma_f32x2_rn_noftz(q_69, scale0_54, a_68);
                        acc[8] = r_70.x;
                        acc[9] = r_70.y;
                        float2 a_71 = make_float2(acc[10], acc[11]);
                        float2 q_72 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                        float2 r_73 = fma_f32x2_rn_noftz(q_72, scale0_54, a_71);
                        acc[10] = r_73.x;
                        acc[11] = r_73.y;
                        float2 a_74 = make_float2(acc[12], acc[13]);
                        float2 q_75 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                        float2 r_76 = fma_f32x2_rn_noftz(q_75, scale0_54, a_74);
                        acc[12] = r_76.x;
                        acc[13] = r_76.y;
                        float2 a_77 = make_float2(acc[14], acc[15]);
                        float2 q_78 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                        float2 r_79 = fma_f32x2_rn_noftz(q_78, scale0_54, a_77);
                        acc[14] = r_79.x;
                        acc[15] = r_79.y;
                        float _unpack_fp4_6[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[2 + 0]));
                            (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(r_4[2 + 0]));
                            #endif
                        }
                        float2 a_80 = make_float2(acc[16], acc[17]);
                        float2 q_81 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                        float2 r_82 = fma_f32x2_rn_noftz(q_81, scale1_55, a_80);
                        acc[16] = r_82.x;
                        acc[17] = r_82.y;
                        float2 a_83 = make_float2(acc[18], acc[19]);
                        float2 q_84 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                        float2 r_85 = fma_f32x2_rn_noftz(q_84, scale1_55, a_83);
                        acc[18] = r_85.x;
                        acc[19] = r_85.y;
                        float2 a_86 = make_float2(acc[20], acc[21]);
                        float2 q_87 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                        float2 r_88 = fma_f32x2_rn_noftz(q_87, scale1_55, a_86);
                        acc[20] = r_88.x;
                        acc[21] = r_88.y;
                        float2 a_89 = make_float2(acc[22], acc[23]);
                        float2 q_90 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                        float2 r_91 = fma_f32x2_rn_noftz(q_90, scale1_55, a_89);
                        acc[22] = r_91.x;
                        acc[23] = r_91.y;
                        float _unpack_fp4_7[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_4[3 + 0]));
                            (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(r_4[3 + 0]));
                            #endif
                        }
                        float2 a_92 = make_float2(acc[24], acc[25]);
                        float2 q_93 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                        float2 r_94 = fma_f32x2_rn_noftz(q_93, scale1_55, a_92);
                        acc[24] = r_94.x;
                        acc[25] = r_94.y;
                        float2 a_95 = make_float2(acc[26], acc[27]);
                        float2 q_96 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                        float2 r_97 = fma_f32x2_rn_noftz(q_96, scale1_55, a_95);
                        acc[26] = r_97.x;
                        acc[27] = r_97.y;
                        float2 a_98 = make_float2(acc[28], acc[29]);
                        float2 q_99 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                        float2 r_100 = fma_f32x2_rn_noftz(q_99, scale1_55, a_98);
                        acc[28] = r_100.x;
                        acc[29] = r_100.y;
                        float2 a_101 = make_float2(acc[30], acc[31]);
                        float2 q_102 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                        float2 r_103 = fma_f32x2_rn_noftz(q_102, scale1_55, a_101);
                        acc[30] = r_103.x;
                        acc[31] = r_103.y;
                        unsigned int t[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[0])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 3))));
                        float2 a2 = make_float2(acc[0], acc[1]);
                        float2 z = add_f32x2_noftz(a2, make_float2(__uint_as_float(t[0] << 16), __uint_as_float(t[0] & 4294901760u)));
                        acc[0] = z.x;
                        acc[1] = z.y;
                        float2 ss2 = fma_f32x2_rn_noftz(z, z, make_float2(ssx, ssy));
                        ssx = ss2.x;
                        ssy = ss2.y;
                        float2 a2_104 = make_float2(acc[2], acc[3]);
                        float2 z_105 = add_f32x2_noftz(a2_104, make_float2(__uint_as_float(t[1] << 16), __uint_as_float(t[1] & 4294901760u)));
                        acc[2] = z_105.x;
                        acc[3] = z_105.y;
                        float2 ss2_106 = fma_f32x2_rn_noftz(z_105, z_105, make_float2(ssx, ssy));
                        ssx = ss2_106.x;
                        ssy = ss2_106.y;
                        float2 a2_107 = make_float2(acc[4], acc[5]);
                        float2 z_108 = add_f32x2_noftz(a2_107, make_float2(__uint_as_float(t[2] << 16), __uint_as_float(t[2] & 4294901760u)));
                        acc[4] = z_108.x;
                        acc[5] = z_108.y;
                        float2 ss2_109 = fma_f32x2_rn_noftz(z_108, z_108, make_float2(ssx, ssy));
                        ssx = ss2_109.x;
                        ssy = ss2_109.y;
                        float2 a2_110 = make_float2(acc[6], acc[7]);
                        float2 z_111 = add_f32x2_noftz(a2_110, make_float2(__uint_as_float(t[3] << 16), __uint_as_float(t[3] & 4294901760u)));
                        acc[6] = z_111.x;
                        acc[7] = z_111.y;
                        float2 ss2_112 = fma_f32x2_rn_noftz(z_111, z_111, make_float2(ssx, ssy));
                        ssx = ss2_112.x;
                        ssy = ss2_112.y;
                        float2 a2_113 = make_float2(acc[8], acc[9]);
                        float2 z_114 = add_f32x2_noftz(a2_113, make_float2(__uint_as_float(t[4] << 16), __uint_as_float(t[4] & 4294901760u)));
                        acc[8] = z_114.x;
                        acc[9] = z_114.y;
                        float2 ss2_115 = fma_f32x2_rn_noftz(z_114, z_114, make_float2(ssx, ssy));
                        ssx = ss2_115.x;
                        ssy = ss2_115.y;
                        float2 a2_116 = make_float2(acc[10], acc[11]);
                        float2 z_117 = add_f32x2_noftz(a2_116, make_float2(__uint_as_float(t[5] << 16), __uint_as_float(t[5] & 4294901760u)));
                        acc[10] = z_117.x;
                        acc[11] = z_117.y;
                        float2 ss2_118 = fma_f32x2_rn_noftz(z_117, z_117, make_float2(ssx, ssy));
                        ssx = ss2_118.x;
                        ssy = ss2_118.y;
                        float2 a2_119 = make_float2(acc[12], acc[13]);
                        float2 z_120 = add_f32x2_noftz(a2_119, make_float2(__uint_as_float(t[6] << 16), __uint_as_float(t[6] & 4294901760u)));
                        acc[12] = z_120.x;
                        acc[13] = z_120.y;
                        float2 ss2_121 = fma_f32x2_rn_noftz(z_120, z_120, make_float2(ssx, ssy));
                        ssx = ss2_121.x;
                        ssy = ss2_121.y;
                        float2 a2_122 = make_float2(acc[14], acc[15]);
                        float2 z_123 = add_f32x2_noftz(a2_122, make_float2(__uint_as_float(t[7] << 16), __uint_as_float(t[7] & 4294901760u)));
                        acc[14] = z_123.x;
                        acc[15] = z_123.y;
                        float2 ss2_124 = fma_f32x2_rn_noftz(z_123, z_123, make_float2(ssx, ssy));
                        ssx = ss2_124.x;
                        ssy = ss2_124.y;
                        float2 a2_125 = make_float2(acc[16], acc[17]);
                        float2 z_126 = add_f32x2_noftz(a2_125, make_float2(__uint_as_float(t[8] << 16), __uint_as_float(t[8] & 4294901760u)));
                        acc[16] = z_126.x;
                        acc[17] = z_126.y;
                        float2 ss2_127 = fma_f32x2_rn_noftz(z_126, z_126, make_float2(ssx, ssy));
                        ssx = ss2_127.x;
                        ssy = ss2_127.y;
                        float2 a2_128 = make_float2(acc[18], acc[19]);
                        float2 z_129 = add_f32x2_noftz(a2_128, make_float2(__uint_as_float(t[9] << 16), __uint_as_float(t[9] & 4294901760u)));
                        acc[18] = z_129.x;
                        acc[19] = z_129.y;
                        float2 ss2_130 = fma_f32x2_rn_noftz(z_129, z_129, make_float2(ssx, ssy));
                        ssx = ss2_130.x;
                        ssy = ss2_130.y;
                        float2 a2_131 = make_float2(acc[20], acc[21]);
                        float2 z_132 = add_f32x2_noftz(a2_131, make_float2(__uint_as_float(t[10] << 16), __uint_as_float(t[10] & 4294901760u)));
                        acc[20] = z_132.x;
                        acc[21] = z_132.y;
                        float2 ss2_133 = fma_f32x2_rn_noftz(z_132, z_132, make_float2(ssx, ssy));
                        ssx = ss2_133.x;
                        ssy = ss2_133.y;
                        float2 a2_134 = make_float2(acc[22], acc[23]);
                        float2 z_135 = add_f32x2_noftz(a2_134, make_float2(__uint_as_float(t[11] << 16), __uint_as_float(t[11] & 4294901760u)));
                        acc[22] = z_135.x;
                        acc[23] = z_135.y;
                        float2 ss2_136 = fma_f32x2_rn_noftz(z_135, z_135, make_float2(ssx, ssy));
                        ssx = ss2_136.x;
                        ssy = ss2_136.y;
                        float2 a2_137 = make_float2(acc[24], acc[25]);
                        float2 z_138 = add_f32x2_noftz(a2_137, make_float2(__uint_as_float(t[12] << 16), __uint_as_float(t[12] & 4294901760u)));
                        acc[24] = z_138.x;
                        acc[25] = z_138.y;
                        float2 ss2_139 = fma_f32x2_rn_noftz(z_138, z_138, make_float2(ssx, ssy));
                        ssx = ss2_139.x;
                        ssy = ss2_139.y;
                        float2 a2_140 = make_float2(acc[26], acc[27]);
                        float2 z_141 = add_f32x2_noftz(a2_140, make_float2(__uint_as_float(t[13] << 16), __uint_as_float(t[13] & 4294901760u)));
                        acc[26] = z_141.x;
                        acc[27] = z_141.y;
                        float2 ss2_142 = fma_f32x2_rn_noftz(z_141, z_141, make_float2(ssx, ssy));
                        ssx = ss2_142.x;
                        ssy = ss2_142.y;
                        float2 a2_143 = make_float2(acc[28], acc[29]);
                        float2 z_144 = add_f32x2_noftz(a2_143, make_float2(__uint_as_float(t[14] << 16), __uint_as_float(t[14] & 4294901760u)));
                        acc[28] = z_144.x;
                        acc[29] = z_144.y;
                        float2 ss2_145 = fma_f32x2_rn_noftz(z_144, z_144, make_float2(ssx, ssy));
                        ssx = ss2_145.x;
                        ssy = ss2_145.y;
                        float2 a2_146 = make_float2(acc[30], acc[31]);
                        float2 z_147 = add_f32x2_noftz(a2_146, make_float2(__uint_as_float(t[15] << 16), __uint_as_float(t[15] & 4294901760u)));
                        acc[30] = z_147.x;
                        acc[31] = z_147.y;
                        float2 ss2_148 = fma_f32x2_rn_noftz(z_147, z_147, make_float2(ssx, ssy));
                        ssx = ss2_148.x;
                        ssy = ss2_148.y;
                        unsigned int acc_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                            acc_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 0))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 1))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 2))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp0 + 16 * (f ^ 3))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                        unsigned int own_149[4];
                        unsigned int own_sc_150[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_149[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_149[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp1)));
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_150[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp1)));
                        unsigned int a_151[4];
                        unsigned int b_152[4];
                        a_151[0] = ((first) ? own_149[0] : _fp4_wire_values_0[4]);
                        b_152[0] = ((first) ? _fp4_wire_values_0[4] : own_149[0]);
                        a_151[1] = ((first) ? own_149[1] : _fp4_wire_values_0[5]);
                        b_152[1] = ((first) ? _fp4_wire_values_0[5] : own_149[1]);
                        a_151[2] = ((first) ? own_149[2] : _fp4_wire_values_0[6]);
                        b_152[2] = ((first) ? _fp4_wire_values_0[6] : own_149[2]);
                        a_151[3] = ((first) ? own_149[3] : _fp4_wire_values_0[7]);
                        b_152[3] = ((first) ? _fp4_wire_values_0[7] : own_149[3]);
                        unsigned int a_sc_153 = ((first) ? own_sc_150[0] : _fp4_wire_scales_0[1]);
                        unsigned int b_sc_154 = ((first) ? _fp4_wire_scales_0[1] : own_sc_150[0]);
                        float acc_155[32] = {0};
                        bool fhi_156 = (f & 2) != 0;
                        bool fodd_157 = (f & 1) != 0;
                        bool fhi_158 = (f & 2) != 0;
                        unsigned int s_159[4];
                        s_159[0] = a_151[0];
                        s_159[1] = a_151[1];
                        s_159[2] = a_151[2];
                        s_159[3] = a_151[3];
                        if (fodd_157) {
                            s_159[0] = a_151[1];
                            s_159[1] = a_151[0];
                            s_159[2] = a_151[3];
                            s_159[3] = a_151[2];
                        }
                        unsigned int r_160[4];
                        r_160[0] = s_159[0];
                        r_160[1] = s_159[1];
                        r_160[2] = s_159[2];
                        r_160[3] = s_159[3];
                        if (fhi_158) {
                            r_160[0] = s_159[2];
                            r_160[1] = s_159[3];
                            r_160[2] = s_159[0];
                            r_160[3] = s_159[1];
                        }
                        bool fodd_161 = (f & 1) != 0;
                        bool fhi_162 = (f & 2) != 0;
                        unsigned int s_163[4];
                        s_163[0] = b_152[0];
                        s_163[1] = b_152[1];
                        s_163[2] = b_152[2];
                        s_163[3] = b_152[3];
                        if (fodd_161) {
                            s_163[0] = b_152[1];
                            s_163[1] = b_152[0];
                            s_163[2] = b_152[3];
                            s_163[3] = b_152[2];
                        }
                        unsigned int r_164[4];
                        r_164[0] = s_163[0];
                        r_164[1] = s_163[1];
                        r_164[2] = s_163[2];
                        r_164[3] = s_163[3];
                        if (fhi_162) {
                            r_164[0] = s_163[2];
                            r_164[1] = s_163[3];
                            r_164[2] = s_163[0];
                            r_164[3] = s_163[1];
                        }
                        uint32_t _e4m3x2_to_f16x2_8;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(a_sc_153)));
                        float _cvt_f32_f16_16;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                        float _cvt_f32_f16_17;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_16), "f"(rs_a_f));
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_17), "f"(rs_a_f));
                        float lo_165 = _mul_4;
                        float hi_166 = _mul_5;
                        if (fhi_156) {
                            lo_165 = _mul_5;
                            hi_166 = _mul_4;
                        }
                        float2 scale0_167 = make_float2(lo_165, lo_165);
                        float2 scale1_168 = make_float2(hi_166, hi_166);
                        float _unpack_fp4_8[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[0]));
                            (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(r_160[0]));
                            #endif
                        }
                        float2 a_169 = make_float2(acc_155[0], acc_155[1]);
                        float2 q_170 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                        float2 r_171 = fma_f32x2_rn_noftz(q_170, scale0_167, a_169);
                        acc_155[0] = r_171.x;
                        acc_155[1] = r_171.y;
                        float2 a_172 = make_float2(acc_155[2], acc_155[3]);
                        float2 q_173 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                        float2 r_174 = fma_f32x2_rn_noftz(q_173, scale0_167, a_172);
                        acc_155[2] = r_174.x;
                        acc_155[3] = r_174.y;
                        float2 a_175 = make_float2(acc_155[4], acc_155[5]);
                        float2 q_176 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                        float2 r_177 = fma_f32x2_rn_noftz(q_176, scale0_167, a_175);
                        acc_155[4] = r_177.x;
                        acc_155[5] = r_177.y;
                        float2 a_178 = make_float2(acc_155[6], acc_155[7]);
                        float2 q_179 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                        float2 r_180 = fma_f32x2_rn_noftz(q_179, scale0_167, a_178);
                        acc_155[6] = r_180.x;
                        acc_155[7] = r_180.y;
                        float _unpack_fp4_9[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[1 + 0]));
                            (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(r_160[1 + 0]));
                            #endif
                        }
                        float2 a_181 = make_float2(acc_155[8], acc_155[9]);
                        float2 q_182 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                        float2 r_183 = fma_f32x2_rn_noftz(q_182, scale0_167, a_181);
                        acc_155[8] = r_183.x;
                        acc_155[9] = r_183.y;
                        float2 a_184 = make_float2(acc_155[10], acc_155[11]);
                        float2 q_185 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                        float2 r_186 = fma_f32x2_rn_noftz(q_185, scale0_167, a_184);
                        acc_155[10] = r_186.x;
                        acc_155[11] = r_186.y;
                        float2 a_187 = make_float2(acc_155[12], acc_155[13]);
                        float2 q_188 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                        float2 r_189 = fma_f32x2_rn_noftz(q_188, scale0_167, a_187);
                        acc_155[12] = r_189.x;
                        acc_155[13] = r_189.y;
                        float2 a_190 = make_float2(acc_155[14], acc_155[15]);
                        float2 q_191 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                        float2 r_192 = fma_f32x2_rn_noftz(q_191, scale0_167, a_190);
                        acc_155[14] = r_192.x;
                        acc_155[15] = r_192.y;
                        float _unpack_fp4_10[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[2 + 0]));
                            (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(r_160[2 + 0]));
                            #endif
                        }
                        float2 a_193 = make_float2(acc_155[16], acc_155[17]);
                        float2 q_194 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                        float2 r_195 = fma_f32x2_rn_noftz(q_194, scale1_168, a_193);
                        acc_155[16] = r_195.x;
                        acc_155[17] = r_195.y;
                        float2 a_196 = make_float2(acc_155[18], acc_155[19]);
                        float2 q_197 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                        float2 r_198 = fma_f32x2_rn_noftz(q_197, scale1_168, a_196);
                        acc_155[18] = r_198.x;
                        acc_155[19] = r_198.y;
                        float2 a_199 = make_float2(acc_155[20], acc_155[21]);
                        float2 q_200 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                        float2 r_201 = fma_f32x2_rn_noftz(q_200, scale1_168, a_199);
                        acc_155[20] = r_201.x;
                        acc_155[21] = r_201.y;
                        float2 a_202 = make_float2(acc_155[22], acc_155[23]);
                        float2 q_203 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                        float2 r_204 = fma_f32x2_rn_noftz(q_203, scale1_168, a_202);
                        acc_155[22] = r_204.x;
                        acc_155[23] = r_204.y;
                        float _unpack_fp4_11[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_160[3 + 0]));
                            (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(r_160[3 + 0]));
                            #endif
                        }
                        float2 a_205 = make_float2(acc_155[24], acc_155[25]);
                        float2 q_206 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                        float2 r_207 = fma_f32x2_rn_noftz(q_206, scale1_168, a_205);
                        acc_155[24] = r_207.x;
                        acc_155[25] = r_207.y;
                        float2 a_208 = make_float2(acc_155[26], acc_155[27]);
                        float2 q_209 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                        float2 r_210 = fma_f32x2_rn_noftz(q_209, scale1_168, a_208);
                        acc_155[26] = r_210.x;
                        acc_155[27] = r_210.y;
                        float2 a_211 = make_float2(acc_155[28], acc_155[29]);
                        float2 q_212 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                        float2 r_213 = fma_f32x2_rn_noftz(q_212, scale1_168, a_211);
                        acc_155[28] = r_213.x;
                        acc_155[29] = r_213.y;
                        float2 a_214 = make_float2(acc_155[30], acc_155[31]);
                        float2 q_215 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                        float2 r_216 = fma_f32x2_rn_noftz(q_215, scale1_168, a_214);
                        acc_155[30] = r_216.x;
                        acc_155[31] = r_216.y;
                        uint32_t _e4m3x2_to_f16x2_9;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(b_sc_154)));
                        float _cvt_f32_f16_18;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                        float _cvt_f32_f16_19;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                        float _mul_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_18), "f"(rs_b_f));
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_19), "f"(rs_b_f));
                        float lo_217 = _mul_6;
                        float hi_218 = _mul_7;
                        if (fhi_156) {
                            lo_217 = _mul_7;
                            hi_218 = _mul_6;
                        }
                        float2 scale0_219 = make_float2(lo_217, lo_217);
                        float2 scale1_220 = make_float2(hi_218, hi_218);
                        float _unpack_fp4_12[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[0]));
                            (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(r_164[0]));
                            #endif
                        }
                        float2 a_221 = make_float2(acc_155[0], acc_155[1]);
                        float2 q_222 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                        float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_219, a_221);
                        acc_155[0] = r_223.x;
                        acc_155[1] = r_223.y;
                        float2 a_224 = make_float2(acc_155[2], acc_155[3]);
                        float2 q_225 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                        float2 r_226 = fma_f32x2_rn_noftz(q_225, scale0_219, a_224);
                        acc_155[2] = r_226.x;
                        acc_155[3] = r_226.y;
                        float2 a_227 = make_float2(acc_155[4], acc_155[5]);
                        float2 q_228 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                        float2 r_229 = fma_f32x2_rn_noftz(q_228, scale0_219, a_227);
                        acc_155[4] = r_229.x;
                        acc_155[5] = r_229.y;
                        float2 a_230 = make_float2(acc_155[6], acc_155[7]);
                        float2 q_231 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                        float2 r_232 = fma_f32x2_rn_noftz(q_231, scale0_219, a_230);
                        acc_155[6] = r_232.x;
                        acc_155[7] = r_232.y;
                        float _unpack_fp4_13[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[1 + 0]));
                            (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(r_164[1 + 0]));
                            #endif
                        }
                        float2 a_233 = make_float2(acc_155[8], acc_155[9]);
                        float2 q_234 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                        float2 r_235 = fma_f32x2_rn_noftz(q_234, scale0_219, a_233);
                        acc_155[8] = r_235.x;
                        acc_155[9] = r_235.y;
                        float2 a_236 = make_float2(acc_155[10], acc_155[11]);
                        float2 q_237 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                        float2 r_238 = fma_f32x2_rn_noftz(q_237, scale0_219, a_236);
                        acc_155[10] = r_238.x;
                        acc_155[11] = r_238.y;
                        float2 a_239 = make_float2(acc_155[12], acc_155[13]);
                        float2 q_240 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                        float2 r_241 = fma_f32x2_rn_noftz(q_240, scale0_219, a_239);
                        acc_155[12] = r_241.x;
                        acc_155[13] = r_241.y;
                        float2 a_242 = make_float2(acc_155[14], acc_155[15]);
                        float2 q_243 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                        float2 r_244 = fma_f32x2_rn_noftz(q_243, scale0_219, a_242);
                        acc_155[14] = r_244.x;
                        acc_155[15] = r_244.y;
                        float _unpack_fp4_14[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[2 + 0]));
                            (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(r_164[2 + 0]));
                            #endif
                        }
                        float2 a_245 = make_float2(acc_155[16], acc_155[17]);
                        float2 q_246 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                        float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_220, a_245);
                        acc_155[16] = r_247.x;
                        acc_155[17] = r_247.y;
                        float2 a_248 = make_float2(acc_155[18], acc_155[19]);
                        float2 q_249 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                        float2 r_250 = fma_f32x2_rn_noftz(q_249, scale1_220, a_248);
                        acc_155[18] = r_250.x;
                        acc_155[19] = r_250.y;
                        float2 a_251 = make_float2(acc_155[20], acc_155[21]);
                        float2 q_252 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                        float2 r_253 = fma_f32x2_rn_noftz(q_252, scale1_220, a_251);
                        acc_155[20] = r_253.x;
                        acc_155[21] = r_253.y;
                        float2 a_254 = make_float2(acc_155[22], acc_155[23]);
                        float2 q_255 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                        float2 r_256 = fma_f32x2_rn_noftz(q_255, scale1_220, a_254);
                        acc_155[22] = r_256.x;
                        acc_155[23] = r_256.y;
                        float _unpack_fp4_15[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_164[3 + 0]));
                            (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(r_164[3 + 0]));
                            #endif
                        }
                        float2 a_257 = make_float2(acc_155[24], acc_155[25]);
                        float2 q_258 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                        float2 r_259 = fma_f32x2_rn_noftz(q_258, scale1_220, a_257);
                        acc_155[24] = r_259.x;
                        acc_155[25] = r_259.y;
                        float2 a_260 = make_float2(acc_155[26], acc_155[27]);
                        float2 q_261 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                        float2 r_262 = fma_f32x2_rn_noftz(q_261, scale1_220, a_260);
                        acc_155[26] = r_262.x;
                        acc_155[27] = r_262.y;
                        float2 a_263 = make_float2(acc_155[28], acc_155[29]);
                        float2 q_264 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                        float2 r_265 = fma_f32x2_rn_noftz(q_264, scale1_220, a_263);
                        acc_155[28] = r_265.x;
                        acc_155[29] = r_265.y;
                        float2 a_266 = make_float2(acc_155[30], acc_155[31]);
                        float2 q_267 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                        float2 r_268 = fma_f32x2_rn_noftz(q_267, scale1_220, a_266);
                        acc_155[30] = r_268.x;
                        acc_155[31] = r_268.y;
                        unsigned int t_269[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_269[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_269[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))));
                        float2 a2_270 = make_float2(acc_155[0], acc_155[1]);
                        float2 z_271 = add_f32x2_noftz(a2_270, make_float2(__uint_as_float(t_269[0] << 16), __uint_as_float(t_269[0] & 4294901760u)));
                        acc_155[0] = z_271.x;
                        acc_155[1] = z_271.y;
                        float2 ss2_272 = fma_f32x2_rn_noftz(z_271, z_271, make_float2(ssx, ssy));
                        ssx = ss2_272.x;
                        ssy = ss2_272.y;
                        float2 a2_273 = make_float2(acc_155[2], acc_155[3]);
                        float2 z_274 = add_f32x2_noftz(a2_273, make_float2(__uint_as_float(t_269[1] << 16), __uint_as_float(t_269[1] & 4294901760u)));
                        acc_155[2] = z_274.x;
                        acc_155[3] = z_274.y;
                        float2 ss2_275 = fma_f32x2_rn_noftz(z_274, z_274, make_float2(ssx, ssy));
                        ssx = ss2_275.x;
                        ssy = ss2_275.y;
                        float2 a2_276 = make_float2(acc_155[4], acc_155[5]);
                        float2 z_277 = add_f32x2_noftz(a2_276, make_float2(__uint_as_float(t_269[2] << 16), __uint_as_float(t_269[2] & 4294901760u)));
                        acc_155[4] = z_277.x;
                        acc_155[5] = z_277.y;
                        float2 ss2_278 = fma_f32x2_rn_noftz(z_277, z_277, make_float2(ssx, ssy));
                        ssx = ss2_278.x;
                        ssy = ss2_278.y;
                        float2 a2_279 = make_float2(acc_155[6], acc_155[7]);
                        float2 z_280 = add_f32x2_noftz(a2_279, make_float2(__uint_as_float(t_269[3] << 16), __uint_as_float(t_269[3] & 4294901760u)));
                        acc_155[6] = z_280.x;
                        acc_155[7] = z_280.y;
                        float2 ss2_281 = fma_f32x2_rn_noftz(z_280, z_280, make_float2(ssx, ssy));
                        ssx = ss2_281.x;
                        ssy = ss2_281.y;
                        float2 a2_282 = make_float2(acc_155[8], acc_155[9]);
                        float2 z_283 = add_f32x2_noftz(a2_282, make_float2(__uint_as_float(t_269[4] << 16), __uint_as_float(t_269[4] & 4294901760u)));
                        acc_155[8] = z_283.x;
                        acc_155[9] = z_283.y;
                        float2 ss2_284 = fma_f32x2_rn_noftz(z_283, z_283, make_float2(ssx, ssy));
                        ssx = ss2_284.x;
                        ssy = ss2_284.y;
                        float2 a2_285 = make_float2(acc_155[10], acc_155[11]);
                        float2 z_286 = add_f32x2_noftz(a2_285, make_float2(__uint_as_float(t_269[5] << 16), __uint_as_float(t_269[5] & 4294901760u)));
                        acc_155[10] = z_286.x;
                        acc_155[11] = z_286.y;
                        float2 ss2_287 = fma_f32x2_rn_noftz(z_286, z_286, make_float2(ssx, ssy));
                        ssx = ss2_287.x;
                        ssy = ss2_287.y;
                        float2 a2_288 = make_float2(acc_155[12], acc_155[13]);
                        float2 z_289 = add_f32x2_noftz(a2_288, make_float2(__uint_as_float(t_269[6] << 16), __uint_as_float(t_269[6] & 4294901760u)));
                        acc_155[12] = z_289.x;
                        acc_155[13] = z_289.y;
                        float2 ss2_290 = fma_f32x2_rn_noftz(z_289, z_289, make_float2(ssx, ssy));
                        ssx = ss2_290.x;
                        ssy = ss2_290.y;
                        float2 a2_291 = make_float2(acc_155[14], acc_155[15]);
                        float2 z_292 = add_f32x2_noftz(a2_291, make_float2(__uint_as_float(t_269[7] << 16), __uint_as_float(t_269[7] & 4294901760u)));
                        acc_155[14] = z_292.x;
                        acc_155[15] = z_292.y;
                        float2 ss2_293 = fma_f32x2_rn_noftz(z_292, z_292, make_float2(ssx, ssy));
                        ssx = ss2_293.x;
                        ssy = ss2_293.y;
                        float2 a2_294 = make_float2(acc_155[16], acc_155[17]);
                        float2 z_295 = add_f32x2_noftz(a2_294, make_float2(__uint_as_float(t_269[8] << 16), __uint_as_float(t_269[8] & 4294901760u)));
                        acc_155[16] = z_295.x;
                        acc_155[17] = z_295.y;
                        float2 ss2_296 = fma_f32x2_rn_noftz(z_295, z_295, make_float2(ssx, ssy));
                        ssx = ss2_296.x;
                        ssy = ss2_296.y;
                        float2 a2_297 = make_float2(acc_155[18], acc_155[19]);
                        float2 z_298 = add_f32x2_noftz(a2_297, make_float2(__uint_as_float(t_269[9] << 16), __uint_as_float(t_269[9] & 4294901760u)));
                        acc_155[18] = z_298.x;
                        acc_155[19] = z_298.y;
                        float2 ss2_299 = fma_f32x2_rn_noftz(z_298, z_298, make_float2(ssx, ssy));
                        ssx = ss2_299.x;
                        ssy = ss2_299.y;
                        float2 a2_300 = make_float2(acc_155[20], acc_155[21]);
                        float2 z_301 = add_f32x2_noftz(a2_300, make_float2(__uint_as_float(t_269[10] << 16), __uint_as_float(t_269[10] & 4294901760u)));
                        acc_155[20] = z_301.x;
                        acc_155[21] = z_301.y;
                        float2 ss2_302 = fma_f32x2_rn_noftz(z_301, z_301, make_float2(ssx, ssy));
                        ssx = ss2_302.x;
                        ssy = ss2_302.y;
                        float2 a2_303 = make_float2(acc_155[22], acc_155[23]);
                        float2 z_304 = add_f32x2_noftz(a2_303, make_float2(__uint_as_float(t_269[11] << 16), __uint_as_float(t_269[11] & 4294901760u)));
                        acc_155[22] = z_304.x;
                        acc_155[23] = z_304.y;
                        float2 ss2_305 = fma_f32x2_rn_noftz(z_304, z_304, make_float2(ssx, ssy));
                        ssx = ss2_305.x;
                        ssy = ss2_305.y;
                        float2 a2_306 = make_float2(acc_155[24], acc_155[25]);
                        float2 z_307 = add_f32x2_noftz(a2_306, make_float2(__uint_as_float(t_269[12] << 16), __uint_as_float(t_269[12] & 4294901760u)));
                        acc_155[24] = z_307.x;
                        acc_155[25] = z_307.y;
                        float2 ss2_308 = fma_f32x2_rn_noftz(z_307, z_307, make_float2(ssx, ssy));
                        ssx = ss2_308.x;
                        ssy = ss2_308.y;
                        float2 a2_309 = make_float2(acc_155[26], acc_155[27]);
                        float2 z_310 = add_f32x2_noftz(a2_309, make_float2(__uint_as_float(t_269[13] << 16), __uint_as_float(t_269[13] & 4294901760u)));
                        acc_155[26] = z_310.x;
                        acc_155[27] = z_310.y;
                        float2 ss2_311 = fma_f32x2_rn_noftz(z_310, z_310, make_float2(ssx, ssy));
                        ssx = ss2_311.x;
                        ssy = ss2_311.y;
                        float2 a2_312 = make_float2(acc_155[28], acc_155[29]);
                        float2 z_313 = add_f32x2_noftz(a2_312, make_float2(__uint_as_float(t_269[14] << 16), __uint_as_float(t_269[14] & 4294901760u)));
                        acc_155[28] = z_313.x;
                        acc_155[29] = z_313.y;
                        float2 ss2_314 = fma_f32x2_rn_noftz(z_313, z_313, make_float2(ssx, ssy));
                        ssx = ss2_314.x;
                        ssy = ss2_314.y;
                        float2 a2_315 = make_float2(acc_155[30], acc_155[31]);
                        float2 z_316 = add_f32x2_noftz(a2_315, make_float2(__uint_as_float(t_269[15] << 16), __uint_as_float(t_269[15] & 4294901760u)));
                        acc_155[30] = z_316.x;
                        acc_155[31] = z_316.y;
                        float2 ss2_317 = fma_f32x2_rn_noftz(z_316, z_316, make_float2(ssx, ssy));
                        ssx = ss2_317.x;
                        ssy = ss2_317.y;
                        unsigned int acc_155_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_155[_lp*2 + 0], acc_155[_lp*2+1 + 0]));
                            acc_155_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))), "r"(acc_155_bf16[0]), "r"(acc_155_bf16[1]), "r"(acc_155_bf16[2]), "r"(acc_155_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))), "r"(acc_155_bf16[4]), "r"(acc_155_bf16[5]), "r"(acc_155_bf16[6]), "r"(acc_155_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))), "r"(acc_155_bf16[8]), "r"(acc_155_bf16[9]), "r"(acc_155_bf16[10]), "r"(acc_155_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))), "r"(acc_155_bf16[12]), "r"(acc_155_bf16[13]), "r"(acc_155_bf16[14]), "r"(acc_155_bf16[15]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_145 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(res_out) + (drow * 12288));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_145), "r"(wsm_addr + (unsigned int)rb_off), "r"((uint32_t)(12288))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    float ss;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ss) : "f"(ssx), "f"(ssy));
                    float _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                    float _add_146;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_146) : "f"(ss), "f"(_shfl_xor_sync_147));
                    ss = _add_146;
                    float _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                    float _add_148;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_148) : "f"(ss), "f"(_shfl_xor_sync_149));
                    ss = _add_148;
                    float _shfl_xor_sync_151 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                    float _add_150;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_150) : "f"(ss), "f"(_shfl_xor_sync_151));
                    ss = _add_150;
                    float _shfl_xor_sync_153 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                    float _add_152;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_152) : "f"(ss), "f"(_shfl_xor_sync_153));
                    ss = _add_152;
                    float _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                    float _add_154;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_154) : "f"(ss), "f"(_shfl_xor_sync_155));
                    ss = _add_154;
                    float _fdiv_156 = __fdiv_rn(ss, 6144.0f);
                    float arg;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(arg) : "f"(_fdiv_156), "f"(eps));
                    float rinv;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rinv) : "f"(arg));
                    float2 r2 = make_float2(rinv, rinv);
                    float2 b2 = make_float2(wbias, wbias);
                    float six = 6.0f;
                    float e4m3_max = 448.0f;
                    float rcp6 = approx_rcp(six);
                    float rcp_gs = approx_rcp(gso);
                    float rcp448 = approx_rcp(e4m3_max);
                    unsigned int row32 = (unsigned int)drow;
                    unsigned int tile = row32 >> 7;
                    unsigned int lanepart = ((unsigned int)lane_2 >> 1) * 512;
                    unsigned int intile = row32 << 4 & 496 | row32 >> 3 & 12 | lanepart;
                    long long sf_base = (long long)tile * 49152;
                    #pragma unroll 1
                    for (int j = 0; j < 6; j++) {
                        int pp_1 = lane_2 + 32 * j;
                        unsigned int tx[16];
                        unsigned int tw[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[0])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_1 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[4])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[4])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_1 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[8])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[8])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_1 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[12])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 3))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[12])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_1 + 16 * (f ^ 3))));
                        float y[32];
                        float2 _f32x2_mul_a_157 = make_float2(__uint_as_float(tx[0] << 16), __uint_as_float(tx[0] & 4294901760u));
                        float2 tz;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&r2));
                        float2 tw_0 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[0] << 16), __uint_as_float(tw[0] & 4294901760u)));
                        float2 t_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&tz), "l"(*(const unsigned long long*)&tw_0));
                        y[0] = t_1.x;
                        y[1] = t_1.y;
                        float2 _f32x2_mul_a_158 = make_float2(__uint_as_float(tx[1] << 16), __uint_as_float(tx[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[1] << 16), __uint_as_float(tw[1] & 4294901760u)));
                        float2 t_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_2));
                        y[2] = t_3.x;
                        y[3] = t_3.y;
                        float2 _f32x2_mul_a_159 = make_float2(__uint_as_float(tx[2] << 16), __uint_as_float(tx[2] & 4294901760u));
                        float2 tz_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&r2));
                        float2 tw_5 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[2] << 16), __uint_as_float(tw[2] & 4294901760u)));
                        float2 t_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&tz_4), "l"(*(const unsigned long long*)&tw_5));
                        y[4] = t_6.x;
                        y[5] = t_6.y;
                        float2 _f32x2_mul_a_160 = make_float2(__uint_as_float(tx[3] << 16), __uint_as_float(tx[3] & 4294901760u));
                        float2 tz_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&r2));
                        float2 tw_8 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[3] << 16), __uint_as_float(tw[3] & 4294901760u)));
                        float2 t_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&tz_7), "l"(*(const unsigned long long*)&tw_8));
                        y[6] = t_9.x;
                        y[7] = t_9.y;
                        float2 _f32x2_mul_a_161 = make_float2(__uint_as_float(tx[4] << 16), __uint_as_float(tx[4] & 4294901760u));
                        float2 tz_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_161), "l"(*(const unsigned long long*)&r2));
                        float2 tw_11 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[4] << 16), __uint_as_float(tw[4] & 4294901760u)));
                        float2 t_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&tz_10), "l"(*(const unsigned long long*)&tw_11));
                        y[8] = t_12.x;
                        y[9] = t_12.y;
                        float2 _f32x2_mul_a_162 = make_float2(__uint_as_float(tx[5] << 16), __uint_as_float(tx[5] & 4294901760u));
                        float2 tz_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&r2));
                        float2 tw_14 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[5] << 16), __uint_as_float(tw[5] & 4294901760u)));
                        float2 t_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&tz_13), "l"(*(const unsigned long long*)&tw_14));
                        y[10] = t_15.x;
                        y[11] = t_15.y;
                        float2 _f32x2_mul_a_163 = make_float2(__uint_as_float(tx[6] << 16), __uint_as_float(tx[6] & 4294901760u));
                        float2 tz_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&r2));
                        float2 tw_17 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[6] << 16), __uint_as_float(tw[6] & 4294901760u)));
                        float2 t_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&tz_16), "l"(*(const unsigned long long*)&tw_17));
                        y[12] = t_18.x;
                        y[13] = t_18.y;
                        float2 _f32x2_mul_a_164 = make_float2(__uint_as_float(tx[7] << 16), __uint_as_float(tx[7] & 4294901760u));
                        float2 tz_19;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&r2));
                        float2 tw_20 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[7] << 16), __uint_as_float(tw[7] & 4294901760u)));
                        float2 t_21;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&tz_19), "l"(*(const unsigned long long*)&tw_20));
                        y[14] = t_21.x;
                        y[15] = t_21.y;
                        float2 _f32x2_mul_a_165 = make_float2(__uint_as_float(tx[8] << 16), __uint_as_float(tx[8] & 4294901760u));
                        float2 tz_22;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&r2));
                        float2 tw_23 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[8] << 16), __uint_as_float(tw[8] & 4294901760u)));
                        float2 t_24;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&tz_22), "l"(*(const unsigned long long*)&tw_23));
                        y[16] = t_24.x;
                        y[17] = t_24.y;
                        float2 _f32x2_mul_a_166 = make_float2(__uint_as_float(tx[9] << 16), __uint_as_float(tx[9] & 4294901760u));
                        float2 tz_25;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&r2));
                        float2 tw_26 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[9] << 16), __uint_as_float(tw[9] & 4294901760u)));
                        float2 t_27;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&tz_25), "l"(*(const unsigned long long*)&tw_26));
                        y[18] = t_27.x;
                        y[19] = t_27.y;
                        float2 _f32x2_mul_a_167 = make_float2(__uint_as_float(tx[10] << 16), __uint_as_float(tx[10] & 4294901760u));
                        float2 tz_28;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_167), "l"(*(const unsigned long long*)&r2));
                        float2 tw_29 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[10] << 16), __uint_as_float(tw[10] & 4294901760u)));
                        float2 t_30;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&tz_28), "l"(*(const unsigned long long*)&tw_29));
                        y[20] = t_30.x;
                        y[21] = t_30.y;
                        float2 _f32x2_mul_a_168 = make_float2(__uint_as_float(tx[11] << 16), __uint_as_float(tx[11] & 4294901760u));
                        float2 tz_31;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&r2));
                        float2 tw_32 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[11] << 16), __uint_as_float(tw[11] & 4294901760u)));
                        float2 t_33;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_33) : "l"(*(const unsigned long long*)&tz_31), "l"(*(const unsigned long long*)&tw_32));
                        y[22] = t_33.x;
                        y[23] = t_33.y;
                        float2 _f32x2_mul_a_169 = make_float2(__uint_as_float(tx[12] << 16), __uint_as_float(tx[12] & 4294901760u));
                        float2 tz_34;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&r2));
                        float2 tw_35 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[12] << 16), __uint_as_float(tw[12] & 4294901760u)));
                        float2 t_36;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_36) : "l"(*(const unsigned long long*)&tz_34), "l"(*(const unsigned long long*)&tw_35));
                        y[24] = t_36.x;
                        y[25] = t_36.y;
                        float2 _f32x2_mul_a_170 = make_float2(__uint_as_float(tx[13] << 16), __uint_as_float(tx[13] & 4294901760u));
                        float2 tz_37;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&r2));
                        float2 tw_38 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[13] << 16), __uint_as_float(tw[13] & 4294901760u)));
                        float2 t_39;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_39) : "l"(*(const unsigned long long*)&tz_37), "l"(*(const unsigned long long*)&tw_38));
                        y[26] = t_39.x;
                        y[27] = t_39.y;
                        float2 _f32x2_mul_a_171 = make_float2(__uint_as_float(tx[14] << 16), __uint_as_float(tx[14] & 4294901760u));
                        float2 tz_40;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&r2));
                        float2 tw_41 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[14] << 16), __uint_as_float(tw[14] & 4294901760u)));
                        float2 t_42;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_42) : "l"(*(const unsigned long long*)&tz_40), "l"(*(const unsigned long long*)&tw_41));
                        y[28] = t_42.x;
                        y[29] = t_42.y;
                        float2 _f32x2_mul_a_172 = make_float2(__uint_as_float(tx[15] << 16), __uint_as_float(tx[15] & 4294901760u));
                        float2 tz_43;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&r2));
                        float2 tw_44 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[15] << 16), __uint_as_float(tw[15] & 4294901760u)));
                        float2 t_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_45) : "l"(*(const unsigned long long*)&tz_43), "l"(*(const unsigned long long*)&tw_44));
                        y[30] = t_45.x;
                        y[31] = t_45.y;
                        unsigned int y_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        long long hoff = sf_base + (long long)(intile + (unsigned int)(8192 * j));
                        float zero = 0.0f;
                        bool fodd_2 = (f & 1) != 0;
                        bool fhi_3 = (f & 2) != 0;
                        float bm[2];
                        unsigned int a_3 = 0;
                        uint32_t _bf16x2_abs_96;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_96) : "r"(y_bf16[0]));
                        uint32_t _bf16x2_max_96;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_96) : "r"(a_3), "r"(_bf16x2_abs_96));
                        a_3 = _bf16x2_max_96;
                        uint32_t _bf16x2_abs_97;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_97) : "r"(y_bf16[1]));
                        uint32_t _bf16x2_max_97;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_97) : "r"(a_3), "r"(_bf16x2_abs_97));
                        a_3 = _bf16x2_max_97;
                        uint32_t _bf16x2_abs_98;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_98) : "r"(y_bf16[2]));
                        uint32_t _bf16x2_max_98;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_98) : "r"(a_3), "r"(_bf16x2_abs_98));
                        a_3 = _bf16x2_max_98;
                        uint32_t _bf16x2_abs_99;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_99) : "r"(y_bf16[3]));
                        uint32_t _bf16x2_max_99;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_99) : "r"(a_3), "r"(_bf16x2_abs_99));
                        a_3 = _bf16x2_max_99;
                        uint32_t _bf16x2_abs_100;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_100) : "r"(y_bf16[4]));
                        uint32_t _bf16x2_max_100;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_100) : "r"(a_3), "r"(_bf16x2_abs_100));
                        a_3 = _bf16x2_max_100;
                        uint32_t _bf16x2_abs_101;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_101) : "r"(y_bf16[5]));
                        uint32_t _bf16x2_max_101;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_101) : "r"(a_3), "r"(_bf16x2_abs_101));
                        a_3 = _bf16x2_max_101;
                        uint32_t _bf16x2_abs_102;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_102) : "r"(y_bf16[6]));
                        uint32_t _bf16x2_max_102;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_102) : "r"(a_3), "r"(_bf16x2_abs_102));
                        a_3 = _bf16x2_max_102;
                        uint32_t _bf16x2_abs_103;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_103) : "r"(y_bf16[7]));
                        uint32_t _bf16x2_max_103;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_103) : "r"(a_3), "r"(_bf16x2_abs_103));
                        a_3 = _bf16x2_max_103;
                        bm[0] = fmaxf(__uint_as_float(a_3 << 16), __uint_as_float(a_3 & 4294901760u));
                        unsigned int a_46_1 = 0;
                        uint32_t _bf16x2_abs_104;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_104) : "r"(y_bf16[8]));
                        uint32_t _bf16x2_max_104;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_104) : "r"(a_46_1), "r"(_bf16x2_abs_104));
                        a_46_1 = _bf16x2_max_104;
                        uint32_t _bf16x2_abs_105;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_105) : "r"(y_bf16[9]));
                        uint32_t _bf16x2_max_105;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_105) : "r"(a_46_1), "r"(_bf16x2_abs_105));
                        a_46_1 = _bf16x2_max_105;
                        uint32_t _bf16x2_abs_106;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_106) : "r"(y_bf16[10]));
                        uint32_t _bf16x2_max_106;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_106) : "r"(a_46_1), "r"(_bf16x2_abs_106));
                        a_46_1 = _bf16x2_max_106;
                        uint32_t _bf16x2_abs_107;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_107) : "r"(y_bf16[11]));
                        uint32_t _bf16x2_max_107;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_107) : "r"(a_46_1), "r"(_bf16x2_abs_107));
                        a_46_1 = _bf16x2_max_107;
                        uint32_t _bf16x2_abs_108;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_108) : "r"(y_bf16[12]));
                        uint32_t _bf16x2_max_108;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_108) : "r"(a_46_1), "r"(_bf16x2_abs_108));
                        a_46_1 = _bf16x2_max_108;
                        uint32_t _bf16x2_abs_109;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_109) : "r"(y_bf16[13]));
                        uint32_t _bf16x2_max_109;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_109) : "r"(a_46_1), "r"(_bf16x2_abs_109));
                        a_46_1 = _bf16x2_max_109;
                        uint32_t _bf16x2_abs_110;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_110) : "r"(y_bf16[14]));
                        uint32_t _bf16x2_max_110;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_110) : "r"(a_46_1), "r"(_bf16x2_abs_110));
                        a_46_1 = _bf16x2_max_110;
                        uint32_t _bf16x2_abs_111;
                        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_111) : "r"(y_bf16[15]));
                        uint32_t _bf16x2_max_111;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_111) : "r"(a_46_1), "r"(_bf16x2_abs_111));
                        a_46_1 = _bf16x2_max_111;
                        bm[1] = fmaxf(__uint_as_float(a_46_1 << 16), __uint_as_float(a_46_1 & 4294901760u));
                        float yb[32];
                        yb[0] = __uint_as_float(y_bf16[0] << 16);
                        yb[1] = __uint_as_float(y_bf16[0] & 4294901760u);
                        yb[2] = __uint_as_float(y_bf16[1] << 16);
                        yb[3] = __uint_as_float(y_bf16[1] & 4294901760u);
                        yb[4] = __uint_as_float(y_bf16[2] << 16);
                        yb[5] = __uint_as_float(y_bf16[2] & 4294901760u);
                        yb[6] = __uint_as_float(y_bf16[3] << 16);
                        yb[7] = __uint_as_float(y_bf16[3] & 4294901760u);
                        yb[8] = __uint_as_float(y_bf16[4] << 16);
                        yb[9] = __uint_as_float(y_bf16[4] & 4294901760u);
                        yb[10] = __uint_as_float(y_bf16[5] << 16);
                        yb[11] = __uint_as_float(y_bf16[5] & 4294901760u);
                        yb[12] = __uint_as_float(y_bf16[6] << 16);
                        yb[13] = __uint_as_float(y_bf16[6] & 4294901760u);
                        yb[14] = __uint_as_float(y_bf16[7] << 16);
                        yb[15] = __uint_as_float(y_bf16[7] & 4294901760u);
                        yb[16] = __uint_as_float(y_bf16[8] << 16);
                        yb[17] = __uint_as_float(y_bf16[8] & 4294901760u);
                        yb[18] = __uint_as_float(y_bf16[9] << 16);
                        yb[19] = __uint_as_float(y_bf16[9] & 4294901760u);
                        yb[20] = __uint_as_float(y_bf16[10] << 16);
                        yb[21] = __uint_as_float(y_bf16[10] & 4294901760u);
                        yb[22] = __uint_as_float(y_bf16[11] << 16);
                        yb[23] = __uint_as_float(y_bf16[11] & 4294901760u);
                        yb[24] = __uint_as_float(y_bf16[12] << 16);
                        yb[25] = __uint_as_float(y_bf16[12] & 4294901760u);
                        yb[26] = __uint_as_float(y_bf16[13] << 16);
                        yb[27] = __uint_as_float(y_bf16[13] & 4294901760u);
                        yb[28] = __uint_as_float(y_bf16[14] << 16);
                        yb[29] = __uint_as_float(y_bf16[14] & 4294901760u);
                        yb[30] = __uint_as_float(y_bf16[15] << 16);
                        yb[31] = __uint_as_float(y_bf16[15] & 4294901760u);
                        float vm0 = bm[0];
                        float vm1 = bm[1];
                        if (fhi_3) {
                            vm0 = bm[1];
                            vm1 = bm[0];
                        }
                        unsigned int sfb[2];
                        float sfv;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv) : "f"(gso), "f"(mul_f32_rn_noftz(vm0, rcp6)));
                        uint16_t _e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(zero), "f"(sfv));
                        sfb[0] = (unsigned int)_e4m3x2_f32_6 & 255;
                        uint32_t _e4m3x2_to_f16x2_10;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(_e4m3x2_f32_6)));
                        float _cvt_f32_f16_20;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                        float _cvt_f32_f16_21;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                        float _rcp_0 = approx_rcp(mul_f32_rn_noftz(_cvt_f32_f16_20, rcp_gs));
                        float oscale = 0.0f;
                        if (vm0 != 0.0f) {
                            oscale = _rcp_0;
                        }
                        float sfv_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv_47) : "f"(gso), "f"(mul_f32_rn_noftz(vm1, rcp6)));
                        uint16_t _e4m3x2_f32_7;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(zero), "f"(sfv_47));
                        sfb[1] = (unsigned int)_e4m3x2_f32_7 & 255;
                        uint32_t _e4m3x2_to_f16x2_11;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(_e4m3x2_f32_7)));
                        float _cvt_f32_f16_22;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                        float _cvt_f32_f16_23;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                        float _rcp_1 = approx_rcp(mul_f32_rn_noftz(_cvt_f32_f16_22, rcp_gs));
                        float oscale_48 = 0.0f;
                        if (vm1 != 0.0f) {
                            oscale_48 = _rcp_1;
                        }
                        float os_p0 = oscale;
                        float os_p1 = oscale_48;
                        if (fhi_3) {
                            os_p0 = oscale_48;
                            os_p1 = oscale;
                        }
                        float2 os2_p0 = make_float2(os_p0, os_p0);
                        float2 os2_p1 = make_float2(os_p1, os_p1);
                        unsigned int w_49[4];
                        float2 _f32x2_mul_a_173 = make_float2(yb[0], yb[1]);
                        float2 g_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_72;
                        uint16_t _e2m1_pair_h_174;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_174) : "f"(g_1.x), "f"(g_1.y));
                        _fp4_pair_72 = (uint32_t)_e2m1_pair_h_174;
                        float2 _f32x2_mul_a_175 = make_float2(yb[2], yb[3]);
                        float2 g_50;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_73;
                        uint16_t _e2m1_pair_h_176;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_176) : "f"(g_50.x), "f"(g_50.y));
                        _fp4_pair_73 = (uint32_t)_e2m1_pair_h_176;
                        float2 _f32x2_mul_a_177 = make_float2(yb[4], yb[5]);
                        float2 g_51;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_177), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_74;
                        uint16_t _e2m1_pair_h_178;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_178) : "f"(g_51.x), "f"(g_51.y));
                        _fp4_pair_74 = (uint32_t)_e2m1_pair_h_178;
                        float2 _f32x2_mul_a_179 = make_float2(yb[6], yb[7]);
                        float2 g_52;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_179), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_e2m1_pack_pair_180;
                        uint16_t _e2m1_pair_h_181;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_181) : "f"(g_52.x), "f"(g_52.y));
                        _fp4_e2m1_pack_pair_180 = (uint32_t)_e2m1_pair_h_181;
                        w_49[0] = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_e2m1_pack_pair_180 << 24;
                        float2 _f32x2_mul_a_182 = make_float2(yb[8], yb[9]);
                        float2 g_53_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_53_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_75;
                        uint16_t _e2m1_pair_h_183;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_183) : "f"(g_53_1.x), "f"(g_53_1.y));
                        _fp4_pair_75 = (uint32_t)_e2m1_pair_h_183;
                        float2 _f32x2_mul_a_184 = make_float2(yb[10], yb[11]);
                        float2 g_54;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_76;
                        uint16_t _e2m1_pair_h_185;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_185) : "f"(g_54.x), "f"(g_54.y));
                        _fp4_pair_76 = (uint32_t)_e2m1_pair_h_185;
                        float2 _f32x2_mul_a_186 = make_float2(yb[12], yb[13]);
                        float2 g_55_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_55_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_77;
                        uint16_t _e2m1_pair_h_187;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_187) : "f"(g_55_1.x), "f"(g_55_1.y));
                        _fp4_pair_77 = (uint32_t)_e2m1_pair_h_187;
                        float2 _f32x2_mul_a_188 = make_float2(yb[14], yb[15]);
                        float2 g_56;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_188), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_e2m1_pack_pair_189;
                        uint16_t _e2m1_pair_h_190;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_190) : "f"(g_56.x), "f"(g_56.y));
                        _fp4_e2m1_pack_pair_189 = (uint32_t)_e2m1_pair_h_190;
                        w_49[1] = _fp4_pair_75 | _fp4_pair_76 << 8 | _fp4_pair_77 << 16 | _fp4_e2m1_pack_pair_189 << 24;
                        float2 _f32x2_mul_a_191 = make_float2(yb[16], yb[17]);
                        float2 g_57_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_57_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_78;
                        uint16_t _e2m1_pair_h_192;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_192) : "f"(g_57_1.x), "f"(g_57_1.y));
                        _fp4_pair_78 = (uint32_t)_e2m1_pair_h_192;
                        float2 _f32x2_mul_a_193 = make_float2(yb[18], yb[19]);
                        float2 g_58;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_79;
                        uint16_t _e2m1_pair_h_194;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_194) : "f"(g_58.x), "f"(g_58.y));
                        _fp4_pair_79 = (uint32_t)_e2m1_pair_h_194;
                        float2 _f32x2_mul_a_195 = make_float2(yb[20], yb[21]);
                        float2 g_59;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_195), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_80;
                        uint16_t _e2m1_pair_h_196;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_196) : "f"(g_59.x), "f"(g_59.y));
                        _fp4_pair_80 = (uint32_t)_e2m1_pair_h_196;
                        float2 _f32x2_mul_a_197 = make_float2(yb[22], yb[23]);
                        float2 g_60;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_e2m1_pack_pair_198;
                        uint16_t _e2m1_pair_h_199;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_199) : "f"(g_60.x), "f"(g_60.y));
                        _fp4_e2m1_pack_pair_198 = (uint32_t)_e2m1_pair_h_199;
                        w_49[2] = _fp4_pair_78 | _fp4_pair_79 << 8 | _fp4_pair_80 << 16 | _fp4_e2m1_pack_pair_198 << 24;
                        float2 _f32x2_mul_a_200 = make_float2(yb[24], yb[25]);
                        float2 g_61;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_81;
                        uint16_t _e2m1_pair_h_201;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_201) : "f"(g_61.x), "f"(g_61.y));
                        _fp4_pair_81 = (uint32_t)_e2m1_pair_h_201;
                        float2 _f32x2_mul_a_202 = make_float2(yb[26], yb[27]);
                        float2 g_62;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_82;
                        uint16_t _e2m1_pair_h_203;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_203) : "f"(g_62.x), "f"(g_62.y));
                        _fp4_pair_82 = (uint32_t)_e2m1_pair_h_203;
                        float2 _f32x2_mul_a_204 = make_float2(yb[28], yb[29]);
                        float2 g_63_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_63_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_83;
                        uint16_t _e2m1_pair_h_205;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_205) : "f"(g_63_1.x), "f"(g_63_1.y));
                        _fp4_pair_83 = (uint32_t)_e2m1_pair_h_205;
                        float2 _f32x2_mul_a_206 = make_float2(yb[30], yb[31]);
                        float2 g_64;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_206), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_e2m1_pack_pair_207;
                        uint16_t _e2m1_pair_h_208;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_208) : "f"(g_64.x), "f"(g_64.y));
                        _fp4_e2m1_pack_pair_207 = (uint32_t)_e2m1_pair_h_208;
                        w_49[3] = _fp4_pair_81 | _fp4_pair_82 << 8 | _fp4_pair_83 << 16 | _fp4_e2m1_pack_pair_207 << 24;
                        unsigned int sw[4];
                        sw[0] = w_49[0];
                        sw[1] = w_49[1];
                        sw[2] = w_49[2];
                        sw[3] = w_49[3];
                        if (fodd_2) {
                            sw[0] = w_49[1];
                            sw[1] = w_49[0];
                            sw[2] = w_49[3];
                            sw[3] = w_49[2];
                        }
                        unsigned int words[4];
                        words[0] = sw[0];
                        words[1] = sw[1];
                        words[2] = sw[2];
                        words[3] = sw[3];
                        if (fhi_3) {
                            words[0] = sw[2];
                            words[1] = sw[3];
                            words[2] = sw[0];
                            words[3] = sw[1];
                        }
                        long long vidx = drow * 3072 + (long long)(16 * pp_1) >> 2;
                        {
                            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(outq) + vidx) = _iv4;
                        }
                        unsigned int half = sfb[0] | sfb[1] << 8;
                        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, half, 1);
                        unsigned int oddmask = -(unsigned int)(lane_2 & 1);
                        unsigned int lo_1 = half & ~oddmask | _shfl_xor_0 & oddmask;
                        unsigned int hi_1 = _shfl_xor_0 & ~oddmask | half & oddmask;
                        unsigned int word_1 = lo_1 | hi_1 << 16;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out_sf + hoff)) + (0)) = word_1;
                    }
                }
                if (n > k + 1) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        long long xrow = (long long)(r0 + warp_1 + 10 * (k + 1));
                        mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 12288);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (xrow * 6144), 12288, xb_full_addr + (warp_1) * 8);
                    }
                }
            }
            if (lane_2 == 0) {
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
                long long pb64 = (long long)pbuf;
                long long cap64_1 = (long long)cap;
                long long peer64_1 = (long long)(1 - rank);
                unsigned int sv4[4] = {2290649224, 2290649224, 2290649224, 2290649224};
                unsigned int ss4[4] = {4294967295, 4294967295, 4294967295, 4294967295};
                unsigned int sr4[4] = {2147483648, 2147483648, 2147483648, 2147483648};
                long long v_base_1 = pb64 * (2 * cap64_1 * 3456) + peer64_1 * cap64_1 * 3072;
                long long s_base_1 = pb64 * (2 * cap64_1 * 3456) + 2 * cap64_1 * 3072 + peer64_1 * cap64_1 * 384;
                long long m_base_1 = pb64 * (2 * cap64_1 * 16) + peer64_1 * cap64_1 * 16;
                long long u = pr64 * 3072 / 16;
                long long lo_2 = u * bid64_1 / g64_1;
                long long hi_2 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_2 + lane64;
                #pragma unroll 1
                for (long long g_2 = g0; g_2 < hi_2; g_2 += 32) {
                    long long widx = v_base_1 + 16 * g_2 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_1_1 = u_0 * bid64_1 / g64_1;
                long long hi_2_1 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_1_1 + lane64;
                #pragma unroll 1
                for (long long g_3 = g0_3; g_3 < hi_2_1; g_3 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_3 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx_1) = _iv4;
                    }
                }
                long long u_4 = pr64 * 16 / 16;
                long long lo_5 = u_4 * bid64_1 / g64_1;
                long long hi_6 = u_4 * (bid64_1 + 1) / g64_1;
                long long g0_7 = lo_5 + lane64;
                #pragma unroll 1
                for (long long g_4 = g0_7; g_4 < hi_6; g_4 += 32) {
                    long long widx_2 = m_base_1 + 16 * g_4 >> 2;
                    {
                        int4 _iv4 = make_int4(sr4[0 + 0], sr4[0 + 1], sr4[0 + 2], sr4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_rs) + widx_2) = _iv4;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
