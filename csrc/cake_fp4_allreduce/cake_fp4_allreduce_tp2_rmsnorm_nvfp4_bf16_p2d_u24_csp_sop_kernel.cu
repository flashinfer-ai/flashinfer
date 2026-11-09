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
#define SMEM_WROW_OFF 1024
#define SMEM_WROW_STAGE_BYTES 12288
#define SMEM_WROW_STRIDE 12288
#define SMEM_CTL_OFF 13312
#define SMEM_CTL_STAGE_BYTES 16
#define SMEM_CTL_STRIDE 16
#define SMEM_WSM_OFF 13376
#define SMEM_WSM_STAGE_BYTES 181632
#define SMEM_WSM_STRIDE 181632
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 181632
#define SMEM_WSM32_STRIDE 181632
#define SMEM_TOTAL 195072
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 1
#ifndef CAKE_FP4_ALLREDUCE_KERNEL
#define CAKE_FP4_ALLREDUCE_KERNEL cake_fp4_allreduce_tp2_rmsnorm_nvfp4_bf16_p2d_u24_csp_sop
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

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
CAKE_FP4_ALLREDUCE_KERNEL(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
    #define rb_full_addr (mbar_base + 64)

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

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // rb_full: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 7) {
        { // workers_main
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 256) {
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
            asm volatile("fence.proxy.async.global;" ::: "memory");
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
                n = (nrows - warp_1 + 8 - 1) / 8;
            }
            if (n > 0 && lane_2 == 0) {
                long long first_row = (long long)(r0 + warp_1);
                mbarrier_arrive_expect_tx(rb_full_addr + (warp_1) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 22704 + 10416), res_in + (first_row * 6144), 12288, rb_full_addr + (warp_1) * 8);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 22704), xq + (first_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
            }
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            asm volatile("barrier.sync 0, 288;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int f = lane_2 >> 1 & 3;
            int xb_off = warp_1 * 22704;
            int own_off = xb_off + 12288;
            int stg_off = xb_off + 10416;
            int rb_off = stg_off;
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
            float gso = 1.0f;
            float gs_me = gs[rank];
            unsigned int _phase_xb_full = 0;
            unsigned int _phase_rb_full = 1;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 8 * k);
                    int ob_off = xb_off + k % 3 * 3472;
#ifdef CAKE_FP4_ALLREDUCE_PREFETCH
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * lane_2)))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 32))))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 64))))));
#endif
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (warp_1) * 8, _phase_xb_full);
                    unsigned int va_0 = wsm_addr + (unsigned int)(ob_off + 16 * lane_2);
                    unsigned int w4_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_0[(0) + 3]))
                        : "r"(va_0));
                    unsigned int c4_0[4];
                    unsigned int carry = (w4_0[0] & 2004318071) + 2004318071;
                    c4_0[0] = w4_0[0] & (carry | 2004318071);
                    unsigned int carry_0 = (w4_0[1] & 2004318071) + 2004318071;
                    c4_0[1] = w4_0[1] & (carry_0 | 2004318071);
                    unsigned int carry_1 = (w4_0[2] & 2004318071) + 2004318071;
                    c4_0[2] = w4_0[2] & (carry_1 | 2004318071);
                    unsigned int carry_2 = (w4_0[3] & 2004318071) + 2004318071;
                    c4_0[3] = w4_0[3] & (carry_2 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_0), "r"(c4_0[0]), "r"(c4_0[1]), "r"(c4_0[2]), "r"(c4_0[3]) : "memory");
                    unsigned int va_1 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_2 + 32));
                    unsigned int w4_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_1[(0) + 3]))
                        : "r"(va_1));
                    unsigned int c4_1[4];
                    unsigned int carry_3 = (w4_1[0] & 2004318071) + 2004318071;
                    c4_1[0] = w4_1[0] & (carry_3 | 2004318071);
                    unsigned int carry_4 = (w4_1[1] & 2004318071) + 2004318071;
                    c4_1[1] = w4_1[1] & (carry_4 | 2004318071);
                    unsigned int carry_5 = (w4_1[2] & 2004318071) + 2004318071;
                    c4_1[2] = w4_1[2] & (carry_5 | 2004318071);
                    unsigned int carry_6 = (w4_1[3] & 2004318071) + 2004318071;
                    c4_1[3] = w4_1[3] & (carry_6 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_1), "r"(c4_1[0]), "r"(c4_1[1]), "r"(c4_1[2]), "r"(c4_1[3]) : "memory");
                    unsigned int va_2 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_2 + 64));
                    unsigned int w4_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_2[(0) + 3]))
                        : "r"(va_2));
                    unsigned int c4_2[4];
                    unsigned int carry_7 = (w4_2[0] & 2004318071) + 2004318071;
                    c4_2[0] = w4_2[0] & (carry_7 | 2004318071);
                    unsigned int carry_8 = (w4_2[1] & 2004318071) + 2004318071;
                    c4_2[1] = w4_2[1] & (carry_8 | 2004318071);
                    unsigned int carry_9 = (w4_2[2] & 2004318071) + 2004318071;
                    c4_2[2] = w4_2[2] & (carry_9 | 2004318071);
                    unsigned int carry_10 = (w4_2[3] & 2004318071) + 2004318071;
                    c4_2[3] = w4_2[3] & (carry_10 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_2), "r"(c4_2[0]), "r"(c4_2[1]), "r"(c4_2[2]), "r"(c4_2[3]) : "memory");
                    unsigned int va_3 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_2 + 96));
                    unsigned int w4_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_3[(0) + 3]))
                        : "r"(va_3));
                    unsigned int c4_3[4];
                    unsigned int carry_11 = (w4_3[0] & 2004318071) + 2004318071;
                    c4_3[0] = w4_3[0] & (carry_11 | 2004318071);
                    unsigned int carry_12 = (w4_3[1] & 2004318071) + 2004318071;
                    c4_3[1] = w4_3[1] & (carry_12 | 2004318071);
                    unsigned int carry_13 = (w4_3[2] & 2004318071) + 2004318071;
                    c4_3[2] = w4_3[2] & (carry_13 | 2004318071);
                    unsigned int carry_14 = (w4_3[3] & 2004318071) + 2004318071;
                    c4_3[3] = w4_3[3] & (carry_14 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_3), "r"(c4_3[0]), "r"(c4_3[1]), "r"(c4_3[2]), "r"(c4_3[3]) : "memory");
                    unsigned int va_4 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_2 + 128));
                    unsigned int w4_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_4[(0) + 3]))
                        : "r"(va_4));
                    unsigned int c4_4[4];
                    unsigned int carry_15 = (w4_4[0] & 2004318071) + 2004318071;
                    c4_4[0] = w4_4[0] & (carry_15 | 2004318071);
                    unsigned int carry_16 = (w4_4[1] & 2004318071) + 2004318071;
                    c4_4[1] = w4_4[1] & (carry_16 | 2004318071);
                    unsigned int carry_17 = (w4_4[2] & 2004318071) + 2004318071;
                    c4_4[2] = w4_4[2] & (carry_17 | 2004318071);
                    unsigned int carry_18 = (w4_4[3] & 2004318071) + 2004318071;
                    c4_4[3] = w4_4[3] & (carry_18 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_4), "r"(c4_4[0]), "r"(c4_4[1]), "r"(c4_4[2]), "r"(c4_4[3]) : "memory");
                    unsigned int va_5 = wsm_addr + (unsigned int)(ob_off + 16 * (lane_2 + 160));
                    unsigned int w4_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&w4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4_5[(0) + 3]))
                        : "r"(va_5));
                    unsigned int c4_5[4];
                    unsigned int carry_19 = (w4_5[0] & 2004318071) + 2004318071;
                    c4_5[0] = w4_5[0] & (carry_19 | 2004318071);
                    unsigned int carry_20 = (w4_5[1] & 2004318071) + 2004318071;
                    c4_5[1] = w4_5[1] & (carry_20 | 2004318071);
                    unsigned int carry_21 = (w4_5[2] & 2004318071) + 2004318071;
                    c4_5[2] = w4_5[2] & (carry_21 | 2004318071);
                    unsigned int carry_22 = (w4_5[3] & 2004318071) + 2004318071;
                    c4_5[3] = w4_5[3] & (carry_22 | 2004318071);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(va_5), "r"(c4_5[0]), "r"(c4_5[1]), "r"(c4_5[2]), "r"(c4_5[3]) : "memory");
                    if (lane_2 < 24) {
                        unsigned int sa = wsm_addr + (unsigned int)(ob_off + 3072 + 16 * lane_2);
                        unsigned int s4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s4[0])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s4[(0) + 3]))
                            : "r"(sa));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sa), "r"(s4[0] & 2139062143), "r"(s4[1] & 2139062143), "r"(s4[2] & 2139062143), "r"(s4[3] & 2139062143) : "memory");
                    }
                    rs = gs_me;
                    if (n > k + 1 && lane_2 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                        long long next_row = (long long)(r0 + warp_1 + 8 * (k + 1));
                        mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(xb_off + (k + 1) % 3 * 3472), xq + (next_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
                    }
                    if (lane_2 < 4) {
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(wsm_addr + (unsigned int)(ob_off + 3072 + 384 + 4 * lane_2));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(rs) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_2 = reinterpret_cast<void*>(peer_wire + (v_base + (rank64 * cap64 + row64) * 3072));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_2), "r"(wsm_addr + (unsigned int)ob_off), "r"((uint32_t)(3072))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_3 = reinterpret_cast<void*>(peer_wire + (s_base + (rank64 * cap64 + row64) * 384));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_3), "r"(wsm_addr + (unsigned int)(ob_off + 3072)), "r"((uint32_t)(384))
                                : "memory");
                        }
                        {
                            void* _cpbulk_dst_4 = reinterpret_cast<void*>(peer_rs + (m_base + (rank64 * cap64 + row64) * 16));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_4), "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 384)), "r"((uint32_t)(16))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                int kd = k - 1;
                if (kd >= 0 && kd < n) {
                    long long drow = (long long)(r0 + warp_1 + 8 * kd);
                    int dob_off = xb_off + kd % 3 * 3472;
                    float own_rs = wsm32[dob_off + 3072 + 384 >> 2];
                    bool first = rank == 0;
                    long long src64 = peer64;
                    long long v_row = v_base + (src64 * cap64 + drow) * 3072;
                    long long s_row = s_base + (src64 * cap64 + drow) * 384;
                    long long m_row = m_base + (src64 * cap64 + drow) * 16;
                    unsigned int rsw[1];
                    uint32_t word;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(word) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = word;
                    float ssx = 0.0f;
                    float ssy = 0.0f;
                    #pragma unroll 1
                    for (int j0 = 0; j0 < 6; j0 += 2) {
                        int pp0 = lane_2 + 32 * j0;
                        int pp1 = pp0 + 32;
                        uint32_t values[8];
                        uint32_t scales[2];
                        uint16_t _wire_h16_5_0;
                        uint16_t _wire_h16_5_1;
                        uint32_t _wire_vpend_5 = 3u;
                        uint32_t _wire_spend_5 = 3u;
                        do {
                            if (_wire_vpend_5 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0)))) : "memory");
                            }
                            if (_wire_vpend_5 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1)))) : "memory");
                            }
                            if (_wire_spend_5 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0)))) : "memory");
                                scales[0] = (uint32_t)_wire_h16_5_0;
                            }
                            if (_wire_spend_5 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1)))) : "memory");
                                scales[1] = (uint32_t)_wire_h16_5_1;
                            }
                            _wire_vpend_5 = ((uint32_t)((((values[0] & ~((values[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[1] & ~((values[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[2] & ~((values[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[3] & ~((values[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((values[4] & ~((values[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[5] & ~((values[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[6] & ~((values[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[7] & ~((values[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_5 = ((uint32_t)(((uint32_t)((scales[0] & 0xFFu) == 0xFFu) | (uint32_t)(((scales[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((scales[1] & 0xFFu) == 0xFFu) | (uint32_t)(((scales[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_5 | _wire_spend_5) != 0u);
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
                            mbarrier_wait(rb_full_addr + (warp_1) * 8, _phase_rb_full);
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
                        unsigned int a[4];
                        unsigned int b[4];
                        a[0] = ((first) ? own[0] : values[0]);
                        b[0] = ((first) ? values[0] : own[0]);
                        a[1] = ((first) ? own[1] : values[1]);
                        b[1] = ((first) ? values[1] : own[1]);
                        a[2] = ((first) ? own[2] : values[2]);
                        b[2] = ((first) ? values[2] : own[2]);
                        a[3] = ((first) ? own[3] : values[3]);
                        b[3] = ((first) ? values[3] : own[3]);
                        unsigned int a_sc = ((first) ? own_sc[0] : scales[0]);
                        unsigned int b_sc = ((first) ? scales[0] : own_sc[0]);
                        float acc[32] = {0};
                        bool fhi = (f & 2) != 0;
                        bool fodd = (f & 1) != 0;
                        bool fhi_0 = (f & 2) != 0;
                        unsigned int s[4];
                        s[0] = a[0];
                        s[1] = a[1];
                        s[2] = a[2];
                        s[3] = a[3];
                        if (fodd) {
                            s[0] = a[1];
                            s[1] = a[0];
                            s[2] = a[3];
                            s[3] = a[2];
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
                        bool fodd_0 = (f & 1) != 0;
                        bool fhi_1 = (f & 2) != 0;
                        unsigned int s_0[4];
                        s_0[0] = b[0];
                        s_0[1] = b[1];
                        s_0[2] = b[2];
                        s_0[3] = b[3];
                        if (fodd_0) {
                            s_0[0] = b[1];
                            s_0[1] = b[0];
                            s_0[2] = b[3];
                            s_0[3] = b[2];
                        }
                        unsigned int r_0[4];
                        r_0[0] = s_0[0];
                        r_0[1] = s_0[1];
                        r_0[2] = s_0[2];
                        r_0[3] = s_0[3];
                        if (fhi_1) {
                            r_0[0] = s_0[2];
                            r_0[1] = s_0[3];
                            r_0[2] = s_0[0];
                            r_0[3] = s_0[1];
                        }
                        uint32_t e4m3x2_to_f16x2;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(a_sc)));
                        float cvt_f32_f16;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                        float cvt_f32_f16_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                        float s0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(cvt_f32_f16), "f"(rs_a_f));
                        float s1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(cvt_f32_f16_1), "f"(rs_a_f));
                        float lo = s0;
                        float hi = s1;
                        if (fhi) {
                            lo = s1;
                            hi = s0;
                        }
                        float2 scale0 = make_float2(lo, lo);
                        float2 scale1 = make_float2(hi, hi);
                        float tv_0[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[0]));
                            (&tv_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_0[0]), "=f"(tv_0[1]), "=f"(tv_0[2]), "=f"(tv_0[3]), "=f"(tv_0[4]), "=f"(tv_0[5]), "=f"(tv_0[6]), "=f"(tv_0[7]) : "r"(r[0]));
                            #endif
                        }
                        float2 a_0 = make_float2(acc[0], acc[1]);
                        float2 q_0 = make_float2(tv_0[0], tv_0[1]);
                        float2 r_1 = fma_f32x2_rn_noftz(q_0, scale0, a_0);
                        acc[0] = r_1.x;
                        acc[1] = r_1.y;
                        float2 a_1 = make_float2(acc[2], acc[3]);
                        float2 q_1 = make_float2(tv_0[2], tv_0[3]);
                        float2 r_2 = fma_f32x2_rn_noftz(q_1, scale0, a_1);
                        acc[2] = r_2.x;
                        acc[3] = r_2.y;
                        float2 a_2 = make_float2(acc[4], acc[5]);
                        float2 q_2 = make_float2(tv_0[4], tv_0[5]);
                        float2 r_3 = fma_f32x2_rn_noftz(q_2, scale0, a_2);
                        acc[4] = r_3.x;
                        acc[5] = r_3.y;
                        float2 a_3 = make_float2(acc[6], acc[7]);
                        float2 q_3 = make_float2(tv_0[6], tv_0[7]);
                        float2 r_4 = fma_f32x2_rn_noftz(q_3, scale0, a_3);
                        acc[6] = r_4.x;
                        acc[7] = r_4.y;
                        float tv_1[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[1 + 0]));
                            (&tv_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_1[0]), "=f"(tv_1[1]), "=f"(tv_1[2]), "=f"(tv_1[3]), "=f"(tv_1[4]), "=f"(tv_1[5]), "=f"(tv_1[6]), "=f"(tv_1[7]) : "r"(r[1 + 0]));
                            #endif
                        }
                        float2 a_4 = make_float2(acc[8], acc[9]);
                        float2 q_4 = make_float2(tv_1[0], tv_1[1]);
                        float2 r_5 = fma_f32x2_rn_noftz(q_4, scale0, a_4);
                        acc[8] = r_5.x;
                        acc[9] = r_5.y;
                        float2 a_5 = make_float2(acc[10], acc[11]);
                        float2 q_5 = make_float2(tv_1[2], tv_1[3]);
                        float2 r_6 = fma_f32x2_rn_noftz(q_5, scale0, a_5);
                        acc[10] = r_6.x;
                        acc[11] = r_6.y;
                        float2 a_6 = make_float2(acc[12], acc[13]);
                        float2 q_6 = make_float2(tv_1[4], tv_1[5]);
                        float2 r_7 = fma_f32x2_rn_noftz(q_6, scale0, a_6);
                        acc[12] = r_7.x;
                        acc[13] = r_7.y;
                        float2 a_7 = make_float2(acc[14], acc[15]);
                        float2 q_7 = make_float2(tv_1[6], tv_1[7]);
                        float2 r_8 = fma_f32x2_rn_noftz(q_7, scale0, a_7);
                        acc[14] = r_8.x;
                        acc[15] = r_8.y;
                        float tv_2[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[2 + 0]));
                            (&tv_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_2[0]), "=f"(tv_2[1]), "=f"(tv_2[2]), "=f"(tv_2[3]), "=f"(tv_2[4]), "=f"(tv_2[5]), "=f"(tv_2[6]), "=f"(tv_2[7]) : "r"(r[2 + 0]));
                            #endif
                        }
                        float2 a_8 = make_float2(acc[16], acc[17]);
                        float2 q_8 = make_float2(tv_2[0], tv_2[1]);
                        float2 r_9 = fma_f32x2_rn_noftz(q_8, scale1, a_8);
                        acc[16] = r_9.x;
                        acc[17] = r_9.y;
                        float2 a_9 = make_float2(acc[18], acc[19]);
                        float2 q_9 = make_float2(tv_2[2], tv_2[3]);
                        float2 r_10 = fma_f32x2_rn_noftz(q_9, scale1, a_9);
                        acc[18] = r_10.x;
                        acc[19] = r_10.y;
                        float2 a_10 = make_float2(acc[20], acc[21]);
                        float2 q_10 = make_float2(tv_2[4], tv_2[5]);
                        float2 r_11 = fma_f32x2_rn_noftz(q_10, scale1, a_10);
                        acc[20] = r_11.x;
                        acc[21] = r_11.y;
                        float2 a_11 = make_float2(acc[22], acc[23]);
                        float2 q_11 = make_float2(tv_2[6], tv_2[7]);
                        float2 r_12 = fma_f32x2_rn_noftz(q_11, scale1, a_11);
                        acc[22] = r_12.x;
                        acc[23] = r_12.y;
                        float tv_3[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r[3 + 0]));
                            (&tv_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_3[0]), "=f"(tv_3[1]), "=f"(tv_3[2]), "=f"(tv_3[3]), "=f"(tv_3[4]), "=f"(tv_3[5]), "=f"(tv_3[6]), "=f"(tv_3[7]) : "r"(r[3 + 0]));
                            #endif
                        }
                        float2 a_12 = make_float2(acc[24], acc[25]);
                        float2 q_12 = make_float2(tv_3[0], tv_3[1]);
                        float2 r_13 = fma_f32x2_rn_noftz(q_12, scale1, a_12);
                        acc[24] = r_13.x;
                        acc[25] = r_13.y;
                        float2 a_13 = make_float2(acc[26], acc[27]);
                        float2 q_13 = make_float2(tv_3[2], tv_3[3]);
                        float2 r_14 = fma_f32x2_rn_noftz(q_13, scale1, a_13);
                        acc[26] = r_14.x;
                        acc[27] = r_14.y;
                        float2 a_14 = make_float2(acc[28], acc[29]);
                        float2 q_14 = make_float2(tv_3[4], tv_3[5]);
                        float2 r_15 = fma_f32x2_rn_noftz(q_14, scale1, a_14);
                        acc[28] = r_15.x;
                        acc[29] = r_15.y;
                        float2 a_15 = make_float2(acc[30], acc[31]);
                        float2 q_15 = make_float2(tv_3[6], tv_3[7]);
                        float2 r_16 = fma_f32x2_rn_noftz(q_15, scale1, a_15);
                        acc[30] = r_16.x;
                        acc[31] = r_16.y;
                        uint32_t e4m3x2_to_f16x2_0;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(b_sc)));
                        float cvt_f32_f16_0;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                        float cvt_f32_f16_1_0;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                        float s0_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(cvt_f32_f16_0), "f"(rs_b_f));
                        float s1_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_0) : "f"(cvt_f32_f16_1_0), "f"(rs_b_f));
                        float lo_0 = s0_0;
                        float hi_0 = s1_0;
                        if (fhi) {
                            lo_0 = s1_0;
                            hi_0 = s0_0;
                        }
                        float2 scale0_0 = make_float2(lo_0, lo_0);
                        float2 scale1_0 = make_float2(hi_0, hi_0);
                        float tv_4[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_0[0]));
                            (&tv_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_4[0]), "=f"(tv_4[1]), "=f"(tv_4[2]), "=f"(tv_4[3]), "=f"(tv_4[4]), "=f"(tv_4[5]), "=f"(tv_4[6]), "=f"(tv_4[7]) : "r"(r_0[0]));
                            #endif
                        }
                        float2 a_16 = make_float2(acc[0], acc[1]);
                        float2 q_16 = make_float2(tv_4[0], tv_4[1]);
                        float2 r_17 = fma_f32x2_rn_noftz(q_16, scale0_0, a_16);
                        acc[0] = r_17.x;
                        acc[1] = r_17.y;
                        float2 a_17 = make_float2(acc[2], acc[3]);
                        float2 q_17 = make_float2(tv_4[2], tv_4[3]);
                        float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0_0, a_17);
                        acc[2] = r_18.x;
                        acc[3] = r_18.y;
                        float2 a_18 = make_float2(acc[4], acc[5]);
                        float2 q_18 = make_float2(tv_4[4], tv_4[5]);
                        float2 r_19 = fma_f32x2_rn_noftz(q_18, scale0_0, a_18);
                        acc[4] = r_19.x;
                        acc[5] = r_19.y;
                        float2 a_19 = make_float2(acc[6], acc[7]);
                        float2 q_19 = make_float2(tv_4[6], tv_4[7]);
                        float2 r_20 = fma_f32x2_rn_noftz(q_19, scale0_0, a_19);
                        acc[6] = r_20.x;
                        acc[7] = r_20.y;
                        float tv_5[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_0[1 + 0]));
                            (&tv_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_5[0]), "=f"(tv_5[1]), "=f"(tv_5[2]), "=f"(tv_5[3]), "=f"(tv_5[4]), "=f"(tv_5[5]), "=f"(tv_5[6]), "=f"(tv_5[7]) : "r"(r_0[1 + 0]));
                            #endif
                        }
                        float2 a_20 = make_float2(acc[8], acc[9]);
                        float2 q_20 = make_float2(tv_5[0], tv_5[1]);
                        float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0_0, a_20);
                        acc[8] = r_21.x;
                        acc[9] = r_21.y;
                        float2 a_21 = make_float2(acc[10], acc[11]);
                        float2 q_21 = make_float2(tv_5[2], tv_5[3]);
                        float2 r_22 = fma_f32x2_rn_noftz(q_21, scale0_0, a_21);
                        acc[10] = r_22.x;
                        acc[11] = r_22.y;
                        float2 a_22 = make_float2(acc[12], acc[13]);
                        float2 q_22 = make_float2(tv_5[4], tv_5[5]);
                        float2 r_23 = fma_f32x2_rn_noftz(q_22, scale0_0, a_22);
                        acc[12] = r_23.x;
                        acc[13] = r_23.y;
                        float2 a_23 = make_float2(acc[14], acc[15]);
                        float2 q_23 = make_float2(tv_5[6], tv_5[7]);
                        float2 r_24 = fma_f32x2_rn_noftz(q_23, scale0_0, a_23);
                        acc[14] = r_24.x;
                        acc[15] = r_24.y;
                        float tv_6[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_0[2 + 0]));
                            (&tv_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_6[0]), "=f"(tv_6[1]), "=f"(tv_6[2]), "=f"(tv_6[3]), "=f"(tv_6[4]), "=f"(tv_6[5]), "=f"(tv_6[6]), "=f"(tv_6[7]) : "r"(r_0[2 + 0]));
                            #endif
                        }
                        float2 a_24 = make_float2(acc[16], acc[17]);
                        float2 q_24 = make_float2(tv_6[0], tv_6[1]);
                        float2 r_25 = fma_f32x2_rn_noftz(q_24, scale1_0, a_24);
                        acc[16] = r_25.x;
                        acc[17] = r_25.y;
                        float2 a_25 = make_float2(acc[18], acc[19]);
                        float2 q_25 = make_float2(tv_6[2], tv_6[3]);
                        float2 r_26 = fma_f32x2_rn_noftz(q_25, scale1_0, a_25);
                        acc[18] = r_26.x;
                        acc[19] = r_26.y;
                        float2 a_26 = make_float2(acc[20], acc[21]);
                        float2 q_26 = make_float2(tv_6[4], tv_6[5]);
                        float2 r_27 = fma_f32x2_rn_noftz(q_26, scale1_0, a_26);
                        acc[20] = r_27.x;
                        acc[21] = r_27.y;
                        float2 a_27 = make_float2(acc[22], acc[23]);
                        float2 q_27 = make_float2(tv_6[6], tv_6[7]);
                        float2 r_28 = fma_f32x2_rn_noftz(q_27, scale1_0, a_27);
                        acc[22] = r_28.x;
                        acc[23] = r_28.y;
                        float tv_7[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_0[3 + 0]));
                            (&tv_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_7[0]), "=f"(tv_7[1]), "=f"(tv_7[2]), "=f"(tv_7[3]), "=f"(tv_7[4]), "=f"(tv_7[5]), "=f"(tv_7[6]), "=f"(tv_7[7]) : "r"(r_0[3 + 0]));
                            #endif
                        }
                        float2 a_28 = make_float2(acc[24], acc[25]);
                        float2 q_28 = make_float2(tv_7[0], tv_7[1]);
                        float2 r_29 = fma_f32x2_rn_noftz(q_28, scale1_0, a_28);
                        acc[24] = r_29.x;
                        acc[25] = r_29.y;
                        float2 a_29 = make_float2(acc[26], acc[27]);
                        float2 q_29 = make_float2(tv_7[2], tv_7[3]);
                        float2 r_30 = fma_f32x2_rn_noftz(q_29, scale1_0, a_29);
                        acc[26] = r_30.x;
                        acc[27] = r_30.y;
                        float2 a_30 = make_float2(acc[28], acc[29]);
                        float2 q_30 = make_float2(tv_7[4], tv_7[5]);
                        float2 r_31 = fma_f32x2_rn_noftz(q_30, scale1_0, a_30);
                        acc[28] = r_31.x;
                        acc[29] = r_31.y;
                        float2 a_31 = make_float2(acc[30], acc[31]);
                        float2 q_31 = make_float2(tv_7[6], tv_7[7]);
                        float2 r_32 = fma_f32x2_rn_noftz(q_31, scale1_0, a_31);
                        acc[30] = r_32.x;
                        acc[31] = r_32.y;
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
                        float2 a2_1_0 = make_float2(acc[0], acc[1]);
                        float2 z_1_0 = add_f32x2_noftz(a2_1_0, make_float2(__uint_as_float(t[0] << 16), __uint_as_float(t[0] & 4294901760u)));
                        acc[0] = z_1_0.x;
                        acc[1] = z_1_0.y;
                        float2 ss2_1_0 = fma_f32x2_rn_noftz(z_1_0, z_1_0, make_float2(ssx, ssy));
                        ssx = ss2_1_0.x;
                        ssy = ss2_1_0.y;
                        float2 a2_1_1 = make_float2(acc[2], acc[3]);
                        float2 z_1_1 = add_f32x2_noftz(a2_1_1, make_float2(__uint_as_float(t[1] << 16), __uint_as_float(t[1] & 4294901760u)));
                        acc[2] = z_1_1.x;
                        acc[3] = z_1_1.y;
                        float2 ss2_1_1 = fma_f32x2_rn_noftz(z_1_1, z_1_1, make_float2(ssx, ssy));
                        ssx = ss2_1_1.x;
                        ssy = ss2_1_1.y;
                        float2 a2_1_2 = make_float2(acc[4], acc[5]);
                        float2 z_1_2 = add_f32x2_noftz(a2_1_2, make_float2(__uint_as_float(t[2] << 16), __uint_as_float(t[2] & 4294901760u)));
                        acc[4] = z_1_2.x;
                        acc[5] = z_1_2.y;
                        float2 ss2_1_2 = fma_f32x2_rn_noftz(z_1_2, z_1_2, make_float2(ssx, ssy));
                        ssx = ss2_1_2.x;
                        ssy = ss2_1_2.y;
                        float2 a2_1_3 = make_float2(acc[6], acc[7]);
                        float2 z_1_3 = add_f32x2_noftz(a2_1_3, make_float2(__uint_as_float(t[3] << 16), __uint_as_float(t[3] & 4294901760u)));
                        acc[6] = z_1_3.x;
                        acc[7] = z_1_3.y;
                        float2 ss2_1_3 = fma_f32x2_rn_noftz(z_1_3, z_1_3, make_float2(ssx, ssy));
                        ssx = ss2_1_3.x;
                        ssy = ss2_1_3.y;
                        float2 a2_1_4 = make_float2(acc[8], acc[9]);
                        float2 z_1_4 = add_f32x2_noftz(a2_1_4, make_float2(__uint_as_float(t[4] << 16), __uint_as_float(t[4] & 4294901760u)));
                        acc[8] = z_1_4.x;
                        acc[9] = z_1_4.y;
                        float2 ss2_1_4 = fma_f32x2_rn_noftz(z_1_4, z_1_4, make_float2(ssx, ssy));
                        ssx = ss2_1_4.x;
                        ssy = ss2_1_4.y;
                        float2 a2_1_5 = make_float2(acc[10], acc[11]);
                        float2 z_1_5 = add_f32x2_noftz(a2_1_5, make_float2(__uint_as_float(t[5] << 16), __uint_as_float(t[5] & 4294901760u)));
                        acc[10] = z_1_5.x;
                        acc[11] = z_1_5.y;
                        float2 ss2_1_5 = fma_f32x2_rn_noftz(z_1_5, z_1_5, make_float2(ssx, ssy));
                        ssx = ss2_1_5.x;
                        ssy = ss2_1_5.y;
                        float2 a2_1_6 = make_float2(acc[12], acc[13]);
                        float2 z_1_6 = add_f32x2_noftz(a2_1_6, make_float2(__uint_as_float(t[6] << 16), __uint_as_float(t[6] & 4294901760u)));
                        acc[12] = z_1_6.x;
                        acc[13] = z_1_6.y;
                        float2 ss2_1_6 = fma_f32x2_rn_noftz(z_1_6, z_1_6, make_float2(ssx, ssy));
                        ssx = ss2_1_6.x;
                        ssy = ss2_1_6.y;
                        float2 a2_1_7 = make_float2(acc[14], acc[15]);
                        float2 z_1_7 = add_f32x2_noftz(a2_1_7, make_float2(__uint_as_float(t[7] << 16), __uint_as_float(t[7] & 4294901760u)));
                        acc[14] = z_1_7.x;
                        acc[15] = z_1_7.y;
                        float2 ss2_1_7 = fma_f32x2_rn_noftz(z_1_7, z_1_7, make_float2(ssx, ssy));
                        ssx = ss2_1_7.x;
                        ssy = ss2_1_7.y;
                        float2 a2_1_8 = make_float2(acc[16], acc[17]);
                        float2 z_1_8 = add_f32x2_noftz(a2_1_8, make_float2(__uint_as_float(t[8] << 16), __uint_as_float(t[8] & 4294901760u)));
                        acc[16] = z_1_8.x;
                        acc[17] = z_1_8.y;
                        float2 ss2_1_8 = fma_f32x2_rn_noftz(z_1_8, z_1_8, make_float2(ssx, ssy));
                        ssx = ss2_1_8.x;
                        ssy = ss2_1_8.y;
                        float2 a2_1_9 = make_float2(acc[18], acc[19]);
                        float2 z_1_9 = add_f32x2_noftz(a2_1_9, make_float2(__uint_as_float(t[9] << 16), __uint_as_float(t[9] & 4294901760u)));
                        acc[18] = z_1_9.x;
                        acc[19] = z_1_9.y;
                        float2 ss2_1_9 = fma_f32x2_rn_noftz(z_1_9, z_1_9, make_float2(ssx, ssy));
                        ssx = ss2_1_9.x;
                        ssy = ss2_1_9.y;
                        float2 a2_1_10 = make_float2(acc[20], acc[21]);
                        float2 z_1_10 = add_f32x2_noftz(a2_1_10, make_float2(__uint_as_float(t[10] << 16), __uint_as_float(t[10] & 4294901760u)));
                        acc[20] = z_1_10.x;
                        acc[21] = z_1_10.y;
                        float2 ss2_1_10 = fma_f32x2_rn_noftz(z_1_10, z_1_10, make_float2(ssx, ssy));
                        ssx = ss2_1_10.x;
                        ssy = ss2_1_10.y;
                        float2 a2_1_11 = make_float2(acc[22], acc[23]);
                        float2 z_1_11 = add_f32x2_noftz(a2_1_11, make_float2(__uint_as_float(t[11] << 16), __uint_as_float(t[11] & 4294901760u)));
                        acc[22] = z_1_11.x;
                        acc[23] = z_1_11.y;
                        float2 ss2_1_11 = fma_f32x2_rn_noftz(z_1_11, z_1_11, make_float2(ssx, ssy));
                        ssx = ss2_1_11.x;
                        ssy = ss2_1_11.y;
                        float2 a2_1_12 = make_float2(acc[24], acc[25]);
                        float2 z_1_12 = add_f32x2_noftz(a2_1_12, make_float2(__uint_as_float(t[12] << 16), __uint_as_float(t[12] & 4294901760u)));
                        acc[24] = z_1_12.x;
                        acc[25] = z_1_12.y;
                        float2 ss2_1_12 = fma_f32x2_rn_noftz(z_1_12, z_1_12, make_float2(ssx, ssy));
                        ssx = ss2_1_12.x;
                        ssy = ss2_1_12.y;
                        float2 a2_1_13 = make_float2(acc[26], acc[27]);
                        float2 z_1_13 = add_f32x2_noftz(a2_1_13, make_float2(__uint_as_float(t[13] << 16), __uint_as_float(t[13] & 4294901760u)));
                        acc[26] = z_1_13.x;
                        acc[27] = z_1_13.y;
                        float2 ss2_1_13 = fma_f32x2_rn_noftz(z_1_13, z_1_13, make_float2(ssx, ssy));
                        ssx = ss2_1_13.x;
                        ssy = ss2_1_13.y;
                        float2 a2_1_14 = make_float2(acc[28], acc[29]);
                        float2 z_1_14 = add_f32x2_noftz(a2_1_14, make_float2(__uint_as_float(t[14] << 16), __uint_as_float(t[14] & 4294901760u)));
                        acc[28] = z_1_14.x;
                        acc[29] = z_1_14.y;
                        float2 ss2_1_14 = fma_f32x2_rn_noftz(z_1_14, z_1_14, make_float2(ssx, ssy));
                        ssx = ss2_1_14.x;
                        ssy = ss2_1_14.y;
                        float2 a2_1_15 = make_float2(acc[30], acc[31]);
                        float2 z_1_15 = add_f32x2_noftz(a2_1_15, make_float2(__uint_as_float(t[15] << 16), __uint_as_float(t[15] & 4294901760u)));
                        acc[30] = z_1_15.x;
                        acc[31] = z_1_15.y;
                        float2 ss2_1_15 = fma_f32x2_rn_noftz(z_1_15, z_1_15, make_float2(ssx, ssy));
                        ssx = ss2_1_15.x;
                        ssy = ss2_1_15.y;
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
                        unsigned int own_0[4];
                        unsigned int own_sc_0[1];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&own_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp1)));
                        asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_0[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp1)));
                        unsigned int a_32[4];
                        unsigned int b_0[4];
                        a_32[0] = ((first) ? own_0[0] : values[4]);
                        b_0[0] = ((first) ? values[4] : own_0[0]);
                        a_32[1] = ((first) ? own_0[1] : values[5]);
                        b_0[1] = ((first) ? values[5] : own_0[1]);
                        a_32[2] = ((first) ? own_0[2] : values[6]);
                        b_0[2] = ((first) ? values[6] : own_0[2]);
                        a_32[3] = ((first) ? own_0[3] : values[7]);
                        b_0[3] = ((first) ? values[7] : own_0[3]);
                        unsigned int a_sc_0 = ((first) ? own_sc_0[0] : scales[1]);
                        unsigned int b_sc_0 = ((first) ? scales[1] : own_sc_0[0]);
                        float acc_0[32] = {0};
                        bool fhi_2 = (f & 2) != 0;
                        bool fodd_1 = (f & 1) != 0;
                        bool fhi_3 = (f & 2) != 0;
                        unsigned int s_1[4];
                        s_1[0] = a_32[0];
                        s_1[1] = a_32[1];
                        s_1[2] = a_32[2];
                        s_1[3] = a_32[3];
                        if (fodd_1) {
                            s_1[0] = a_32[1];
                            s_1[1] = a_32[0];
                            s_1[2] = a_32[3];
                            s_1[3] = a_32[2];
                        }
                        unsigned int r_33[4];
                        r_33[0] = s_1[0];
                        r_33[1] = s_1[1];
                        r_33[2] = s_1[2];
                        r_33[3] = s_1[3];
                        if (fhi_3) {
                            r_33[0] = s_1[2];
                            r_33[1] = s_1[3];
                            r_33[2] = s_1[0];
                            r_33[3] = s_1[1];
                        }
                        bool fodd_2 = (f & 1) != 0;
                        bool fhi_4 = (f & 2) != 0;
                        unsigned int s_2[4];
                        s_2[0] = b_0[0];
                        s_2[1] = b_0[1];
                        s_2[2] = b_0[2];
                        s_2[3] = b_0[3];
                        if (fodd_2) {
                            s_2[0] = b_0[1];
                            s_2[1] = b_0[0];
                            s_2[2] = b_0[3];
                            s_2[3] = b_0[2];
                        }
                        unsigned int r_34[4];
                        r_34[0] = s_2[0];
                        r_34[1] = s_2[1];
                        r_34[2] = s_2[2];
                        r_34[3] = s_2[3];
                        if (fhi_4) {
                            r_34[0] = s_2[2];
                            r_34[1] = s_2[3];
                            r_34[2] = s_2[0];
                            r_34[3] = s_2[1];
                        }
                        uint32_t e4m3x2_to_f16x2_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(a_sc_0)));
                        float cvt_f32_f16_2;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                        float cvt_f32_f16_1_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                        float s0_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(cvt_f32_f16_2), "f"(rs_a_f));
                        float s1_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_1) : "f"(cvt_f32_f16_1_1), "f"(rs_a_f));
                        float lo_1 = s0_1;
                        float hi_1 = s1_1;
                        if (fhi_2) {
                            lo_1 = s1_1;
                            hi_1 = s0_1;
                        }
                        float2 scale0_1 = make_float2(lo_1, lo_1);
                        float2 scale1_1 = make_float2(hi_1, hi_1);
                        float tv_8[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_33[0]));
                            (&tv_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_8[0]), "=f"(tv_8[1]), "=f"(tv_8[2]), "=f"(tv_8[3]), "=f"(tv_8[4]), "=f"(tv_8[5]), "=f"(tv_8[6]), "=f"(tv_8[7]) : "r"(r_33[0]));
                            #endif
                        }
                        float2 a_33 = make_float2(acc_0[0], acc_0[1]);
                        float2 q_32 = make_float2(tv_8[0], tv_8[1]);
                        float2 r_35 = fma_f32x2_rn_noftz(q_32, scale0_1, a_33);
                        acc_0[0] = r_35.x;
                        acc_0[1] = r_35.y;
                        float2 a_34 = make_float2(acc_0[2], acc_0[3]);
                        float2 q_33 = make_float2(tv_8[2], tv_8[3]);
                        float2 r_36 = fma_f32x2_rn_noftz(q_33, scale0_1, a_34);
                        acc_0[2] = r_36.x;
                        acc_0[3] = r_36.y;
                        float2 a_35 = make_float2(acc_0[4], acc_0[5]);
                        float2 q_34 = make_float2(tv_8[4], tv_8[5]);
                        float2 r_37 = fma_f32x2_rn_noftz(q_34, scale0_1, a_35);
                        acc_0[4] = r_37.x;
                        acc_0[5] = r_37.y;
                        float2 a_36 = make_float2(acc_0[6], acc_0[7]);
                        float2 q_35 = make_float2(tv_8[6], tv_8[7]);
                        float2 r_38 = fma_f32x2_rn_noftz(q_35, scale0_1, a_36);
                        acc_0[6] = r_38.x;
                        acc_0[7] = r_38.y;
                        float tv_9[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_33[1 + 0]));
                            (&tv_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_9[0]), "=f"(tv_9[1]), "=f"(tv_9[2]), "=f"(tv_9[3]), "=f"(tv_9[4]), "=f"(tv_9[5]), "=f"(tv_9[6]), "=f"(tv_9[7]) : "r"(r_33[1 + 0]));
                            #endif
                        }
                        float2 a_37 = make_float2(acc_0[8], acc_0[9]);
                        float2 q_36 = make_float2(tv_9[0], tv_9[1]);
                        float2 r_39 = fma_f32x2_rn_noftz(q_36, scale0_1, a_37);
                        acc_0[8] = r_39.x;
                        acc_0[9] = r_39.y;
                        float2 a_38 = make_float2(acc_0[10], acc_0[11]);
                        float2 q_37 = make_float2(tv_9[2], tv_9[3]);
                        float2 r_40 = fma_f32x2_rn_noftz(q_37, scale0_1, a_38);
                        acc_0[10] = r_40.x;
                        acc_0[11] = r_40.y;
                        float2 a_39 = make_float2(acc_0[12], acc_0[13]);
                        float2 q_38 = make_float2(tv_9[4], tv_9[5]);
                        float2 r_41 = fma_f32x2_rn_noftz(q_38, scale0_1, a_39);
                        acc_0[12] = r_41.x;
                        acc_0[13] = r_41.y;
                        float2 a_40 = make_float2(acc_0[14], acc_0[15]);
                        float2 q_39 = make_float2(tv_9[6], tv_9[7]);
                        float2 r_42 = fma_f32x2_rn_noftz(q_39, scale0_1, a_40);
                        acc_0[14] = r_42.x;
                        acc_0[15] = r_42.y;
                        float tv_10[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_33[2 + 0]));
                            (&tv_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_10[0]), "=f"(tv_10[1]), "=f"(tv_10[2]), "=f"(tv_10[3]), "=f"(tv_10[4]), "=f"(tv_10[5]), "=f"(tv_10[6]), "=f"(tv_10[7]) : "r"(r_33[2 + 0]));
                            #endif
                        }
                        float2 a_41 = make_float2(acc_0[16], acc_0[17]);
                        float2 q_40 = make_float2(tv_10[0], tv_10[1]);
                        float2 r_43 = fma_f32x2_rn_noftz(q_40, scale1_1, a_41);
                        acc_0[16] = r_43.x;
                        acc_0[17] = r_43.y;
                        float2 a_42 = make_float2(acc_0[18], acc_0[19]);
                        float2 q_41 = make_float2(tv_10[2], tv_10[3]);
                        float2 r_44 = fma_f32x2_rn_noftz(q_41, scale1_1, a_42);
                        acc_0[18] = r_44.x;
                        acc_0[19] = r_44.y;
                        float2 a_43 = make_float2(acc_0[20], acc_0[21]);
                        float2 q_42 = make_float2(tv_10[4], tv_10[5]);
                        float2 r_45 = fma_f32x2_rn_noftz(q_42, scale1_1, a_43);
                        acc_0[20] = r_45.x;
                        acc_0[21] = r_45.y;
                        float2 a_44 = make_float2(acc_0[22], acc_0[23]);
                        float2 q_43 = make_float2(tv_10[6], tv_10[7]);
                        float2 r_46 = fma_f32x2_rn_noftz(q_43, scale1_1, a_44);
                        acc_0[22] = r_46.x;
                        acc_0[23] = r_46.y;
                        float tv_11[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_33[3 + 0]));
                            (&tv_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_11[0]), "=f"(tv_11[1]), "=f"(tv_11[2]), "=f"(tv_11[3]), "=f"(tv_11[4]), "=f"(tv_11[5]), "=f"(tv_11[6]), "=f"(tv_11[7]) : "r"(r_33[3 + 0]));
                            #endif
                        }
                        float2 a_45 = make_float2(acc_0[24], acc_0[25]);
                        float2 q_44 = make_float2(tv_11[0], tv_11[1]);
                        float2 r_47 = fma_f32x2_rn_noftz(q_44, scale1_1, a_45);
                        acc_0[24] = r_47.x;
                        acc_0[25] = r_47.y;
                        float2 a_46 = make_float2(acc_0[26], acc_0[27]);
                        float2 q_45 = make_float2(tv_11[2], tv_11[3]);
                        float2 r_48 = fma_f32x2_rn_noftz(q_45, scale1_1, a_46);
                        acc_0[26] = r_48.x;
                        acc_0[27] = r_48.y;
                        float2 a_47 = make_float2(acc_0[28], acc_0[29]);
                        float2 q_46 = make_float2(tv_11[4], tv_11[5]);
                        float2 r_49 = fma_f32x2_rn_noftz(q_46, scale1_1, a_47);
                        acc_0[28] = r_49.x;
                        acc_0[29] = r_49.y;
                        float2 a_48 = make_float2(acc_0[30], acc_0[31]);
                        float2 q_47 = make_float2(tv_11[6], tv_11[7]);
                        float2 r_50 = fma_f32x2_rn_noftz(q_47, scale1_1, a_48);
                        acc_0[30] = r_50.x;
                        acc_0[31] = r_50.y;
                        uint32_t e4m3x2_to_f16x2_2;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(b_sc_0)));
                        float cvt_f32_f16_3;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                        float cvt_f32_f16_1_2;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                        float s0_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(cvt_f32_f16_3), "f"(rs_b_f));
                        float s1_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_2) : "f"(cvt_f32_f16_1_2), "f"(rs_b_f));
                        float lo_2 = s0_2;
                        float hi_2 = s1_2;
                        if (fhi_2) {
                            lo_2 = s1_2;
                            hi_2 = s0_2;
                        }
                        float2 scale0_2 = make_float2(lo_2, lo_2);
                        float2 scale1_2 = make_float2(hi_2, hi_2);
                        float tv_12[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_34[0]));
                            (&tv_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_12[0]), "=f"(tv_12[1]), "=f"(tv_12[2]), "=f"(tv_12[3]), "=f"(tv_12[4]), "=f"(tv_12[5]), "=f"(tv_12[6]), "=f"(tv_12[7]) : "r"(r_34[0]));
                            #endif
                        }
                        float2 a_49 = make_float2(acc_0[0], acc_0[1]);
                        float2 q_48 = make_float2(tv_12[0], tv_12[1]);
                        float2 r_51 = fma_f32x2_rn_noftz(q_48, scale0_2, a_49);
                        acc_0[0] = r_51.x;
                        acc_0[1] = r_51.y;
                        float2 a_50 = make_float2(acc_0[2], acc_0[3]);
                        float2 q_49 = make_float2(tv_12[2], tv_12[3]);
                        float2 r_52 = fma_f32x2_rn_noftz(q_49, scale0_2, a_50);
                        acc_0[2] = r_52.x;
                        acc_0[3] = r_52.y;
                        float2 a_51 = make_float2(acc_0[4], acc_0[5]);
                        float2 q_50 = make_float2(tv_12[4], tv_12[5]);
                        float2 r_53 = fma_f32x2_rn_noftz(q_50, scale0_2, a_51);
                        acc_0[4] = r_53.x;
                        acc_0[5] = r_53.y;
                        float2 a_52 = make_float2(acc_0[6], acc_0[7]);
                        float2 q_51 = make_float2(tv_12[6], tv_12[7]);
                        float2 r_54 = fma_f32x2_rn_noftz(q_51, scale0_2, a_52);
                        acc_0[6] = r_54.x;
                        acc_0[7] = r_54.y;
                        float tv_13[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_34[1 + 0]));
                            (&tv_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_13[0]), "=f"(tv_13[1]), "=f"(tv_13[2]), "=f"(tv_13[3]), "=f"(tv_13[4]), "=f"(tv_13[5]), "=f"(tv_13[6]), "=f"(tv_13[7]) : "r"(r_34[1 + 0]));
                            #endif
                        }
                        float2 a_53 = make_float2(acc_0[8], acc_0[9]);
                        float2 q_52 = make_float2(tv_13[0], tv_13[1]);
                        float2 r_55 = fma_f32x2_rn_noftz(q_52, scale0_2, a_53);
                        acc_0[8] = r_55.x;
                        acc_0[9] = r_55.y;
                        float2 a_54 = make_float2(acc_0[10], acc_0[11]);
                        float2 q_53 = make_float2(tv_13[2], tv_13[3]);
                        float2 r_56 = fma_f32x2_rn_noftz(q_53, scale0_2, a_54);
                        acc_0[10] = r_56.x;
                        acc_0[11] = r_56.y;
                        float2 a_55 = make_float2(acc_0[12], acc_0[13]);
                        float2 q_54 = make_float2(tv_13[4], tv_13[5]);
                        float2 r_57 = fma_f32x2_rn_noftz(q_54, scale0_2, a_55);
                        acc_0[12] = r_57.x;
                        acc_0[13] = r_57.y;
                        float2 a_56 = make_float2(acc_0[14], acc_0[15]);
                        float2 q_55 = make_float2(tv_13[6], tv_13[7]);
                        float2 r_58 = fma_f32x2_rn_noftz(q_55, scale0_2, a_56);
                        acc_0[14] = r_58.x;
                        acc_0[15] = r_58.y;
                        float tv_14[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_34[2 + 0]));
                            (&tv_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_14[0]), "=f"(tv_14[1]), "=f"(tv_14[2]), "=f"(tv_14[3]), "=f"(tv_14[4]), "=f"(tv_14[5]), "=f"(tv_14[6]), "=f"(tv_14[7]) : "r"(r_34[2 + 0]));
                            #endif
                        }
                        float2 a_57 = make_float2(acc_0[16], acc_0[17]);
                        float2 q_56 = make_float2(tv_14[0], tv_14[1]);
                        float2 r_59 = fma_f32x2_rn_noftz(q_56, scale1_2, a_57);
                        acc_0[16] = r_59.x;
                        acc_0[17] = r_59.y;
                        float2 a_58 = make_float2(acc_0[18], acc_0[19]);
                        float2 q_57 = make_float2(tv_14[2], tv_14[3]);
                        float2 r_60 = fma_f32x2_rn_noftz(q_57, scale1_2, a_58);
                        acc_0[18] = r_60.x;
                        acc_0[19] = r_60.y;
                        float2 a_59 = make_float2(acc_0[20], acc_0[21]);
                        float2 q_58 = make_float2(tv_14[4], tv_14[5]);
                        float2 r_61 = fma_f32x2_rn_noftz(q_58, scale1_2, a_59);
                        acc_0[20] = r_61.x;
                        acc_0[21] = r_61.y;
                        float2 a_60 = make_float2(acc_0[22], acc_0[23]);
                        float2 q_59 = make_float2(tv_14[6], tv_14[7]);
                        float2 r_62 = fma_f32x2_rn_noftz(q_59, scale1_2, a_60);
                        acc_0[22] = r_62.x;
                        acc_0[23] = r_62.y;
                        float tv_15[8];
                        {
                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                            uint32_t _fp4_decode_pairs[4];
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_34[3 + 0]));
                            (&tv_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                            (&tv_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                            (&tv_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                            (&tv_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                            (&tv_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                            (&tv_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                            (&tv_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                            (&tv_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                            #else
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_15[0]), "=f"(tv_15[1]), "=f"(tv_15[2]), "=f"(tv_15[3]), "=f"(tv_15[4]), "=f"(tv_15[5]), "=f"(tv_15[6]), "=f"(tv_15[7]) : "r"(r_34[3 + 0]));
                            #endif
                        }
                        float2 a_61 = make_float2(acc_0[24], acc_0[25]);
                        float2 q_60 = make_float2(tv_15[0], tv_15[1]);
                        float2 r_63 = fma_f32x2_rn_noftz(q_60, scale1_2, a_61);
                        acc_0[24] = r_63.x;
                        acc_0[25] = r_63.y;
                        float2 a_62 = make_float2(acc_0[26], acc_0[27]);
                        float2 q_61 = make_float2(tv_15[2], tv_15[3]);
                        float2 r_64 = fma_f32x2_rn_noftz(q_61, scale1_2, a_62);
                        acc_0[26] = r_64.x;
                        acc_0[27] = r_64.y;
                        float2 a_63 = make_float2(acc_0[28], acc_0[29]);
                        float2 q_62 = make_float2(tv_15[4], tv_15[5]);
                        float2 r_65 = fma_f32x2_rn_noftz(q_62, scale1_2, a_63);
                        acc_0[28] = r_65.x;
                        acc_0[29] = r_65.y;
                        float2 a_64 = make_float2(acc_0[30], acc_0[31]);
                        float2 q_63 = make_float2(tv_15[6], tv_15[7]);
                        float2 r_66 = fma_f32x2_rn_noftz(q_63, scale1_2, a_64);
                        acc_0[30] = r_66.x;
                        acc_0[31] = r_66.y;
                        unsigned int t_0[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&t_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))));
                        float2 a2_1_16 = make_float2(acc_0[0], acc_0[1]);
                        float2 z_1_16 = add_f32x2_noftz(a2_1_16, make_float2(__uint_as_float(t_0[0] << 16), __uint_as_float(t_0[0] & 4294901760u)));
                        acc_0[0] = z_1_16.x;
                        acc_0[1] = z_1_16.y;
                        float2 ss2_1_16 = fma_f32x2_rn_noftz(z_1_16, z_1_16, make_float2(ssx, ssy));
                        ssx = ss2_1_16.x;
                        ssy = ss2_1_16.y;
                        float2 a2_1_17 = make_float2(acc_0[2], acc_0[3]);
                        float2 z_1_17 = add_f32x2_noftz(a2_1_17, make_float2(__uint_as_float(t_0[1] << 16), __uint_as_float(t_0[1] & 4294901760u)));
                        acc_0[2] = z_1_17.x;
                        acc_0[3] = z_1_17.y;
                        float2 ss2_1_17 = fma_f32x2_rn_noftz(z_1_17, z_1_17, make_float2(ssx, ssy));
                        ssx = ss2_1_17.x;
                        ssy = ss2_1_17.y;
                        float2 a2_1_18 = make_float2(acc_0[4], acc_0[5]);
                        float2 z_1_18 = add_f32x2_noftz(a2_1_18, make_float2(__uint_as_float(t_0[2] << 16), __uint_as_float(t_0[2] & 4294901760u)));
                        acc_0[4] = z_1_18.x;
                        acc_0[5] = z_1_18.y;
                        float2 ss2_1_18 = fma_f32x2_rn_noftz(z_1_18, z_1_18, make_float2(ssx, ssy));
                        ssx = ss2_1_18.x;
                        ssy = ss2_1_18.y;
                        float2 a2_1_19 = make_float2(acc_0[6], acc_0[7]);
                        float2 z_1_19 = add_f32x2_noftz(a2_1_19, make_float2(__uint_as_float(t_0[3] << 16), __uint_as_float(t_0[3] & 4294901760u)));
                        acc_0[6] = z_1_19.x;
                        acc_0[7] = z_1_19.y;
                        float2 ss2_1_19 = fma_f32x2_rn_noftz(z_1_19, z_1_19, make_float2(ssx, ssy));
                        ssx = ss2_1_19.x;
                        ssy = ss2_1_19.y;
                        float2 a2_1_20 = make_float2(acc_0[8], acc_0[9]);
                        float2 z_1_20 = add_f32x2_noftz(a2_1_20, make_float2(__uint_as_float(t_0[4] << 16), __uint_as_float(t_0[4] & 4294901760u)));
                        acc_0[8] = z_1_20.x;
                        acc_0[9] = z_1_20.y;
                        float2 ss2_1_20 = fma_f32x2_rn_noftz(z_1_20, z_1_20, make_float2(ssx, ssy));
                        ssx = ss2_1_20.x;
                        ssy = ss2_1_20.y;
                        float2 a2_1_21 = make_float2(acc_0[10], acc_0[11]);
                        float2 z_1_21 = add_f32x2_noftz(a2_1_21, make_float2(__uint_as_float(t_0[5] << 16), __uint_as_float(t_0[5] & 4294901760u)));
                        acc_0[10] = z_1_21.x;
                        acc_0[11] = z_1_21.y;
                        float2 ss2_1_21 = fma_f32x2_rn_noftz(z_1_21, z_1_21, make_float2(ssx, ssy));
                        ssx = ss2_1_21.x;
                        ssy = ss2_1_21.y;
                        float2 a2_1_22 = make_float2(acc_0[12], acc_0[13]);
                        float2 z_1_22 = add_f32x2_noftz(a2_1_22, make_float2(__uint_as_float(t_0[6] << 16), __uint_as_float(t_0[6] & 4294901760u)));
                        acc_0[12] = z_1_22.x;
                        acc_0[13] = z_1_22.y;
                        float2 ss2_1_22 = fma_f32x2_rn_noftz(z_1_22, z_1_22, make_float2(ssx, ssy));
                        ssx = ss2_1_22.x;
                        ssy = ss2_1_22.y;
                        float2 a2_1_23 = make_float2(acc_0[14], acc_0[15]);
                        float2 z_1_23 = add_f32x2_noftz(a2_1_23, make_float2(__uint_as_float(t_0[7] << 16), __uint_as_float(t_0[7] & 4294901760u)));
                        acc_0[14] = z_1_23.x;
                        acc_0[15] = z_1_23.y;
                        float2 ss2_1_23 = fma_f32x2_rn_noftz(z_1_23, z_1_23, make_float2(ssx, ssy));
                        ssx = ss2_1_23.x;
                        ssy = ss2_1_23.y;
                        float2 a2_1_24 = make_float2(acc_0[16], acc_0[17]);
                        float2 z_1_24 = add_f32x2_noftz(a2_1_24, make_float2(__uint_as_float(t_0[8] << 16), __uint_as_float(t_0[8] & 4294901760u)));
                        acc_0[16] = z_1_24.x;
                        acc_0[17] = z_1_24.y;
                        float2 ss2_1_24 = fma_f32x2_rn_noftz(z_1_24, z_1_24, make_float2(ssx, ssy));
                        ssx = ss2_1_24.x;
                        ssy = ss2_1_24.y;
                        float2 a2_1_25 = make_float2(acc_0[18], acc_0[19]);
                        float2 z_1_25 = add_f32x2_noftz(a2_1_25, make_float2(__uint_as_float(t_0[9] << 16), __uint_as_float(t_0[9] & 4294901760u)));
                        acc_0[18] = z_1_25.x;
                        acc_0[19] = z_1_25.y;
                        float2 ss2_1_25 = fma_f32x2_rn_noftz(z_1_25, z_1_25, make_float2(ssx, ssy));
                        ssx = ss2_1_25.x;
                        ssy = ss2_1_25.y;
                        float2 a2_1_26 = make_float2(acc_0[20], acc_0[21]);
                        float2 z_1_26 = add_f32x2_noftz(a2_1_26, make_float2(__uint_as_float(t_0[10] << 16), __uint_as_float(t_0[10] & 4294901760u)));
                        acc_0[20] = z_1_26.x;
                        acc_0[21] = z_1_26.y;
                        float2 ss2_1_26 = fma_f32x2_rn_noftz(z_1_26, z_1_26, make_float2(ssx, ssy));
                        ssx = ss2_1_26.x;
                        ssy = ss2_1_26.y;
                        float2 a2_1_27 = make_float2(acc_0[22], acc_0[23]);
                        float2 z_1_27 = add_f32x2_noftz(a2_1_27, make_float2(__uint_as_float(t_0[11] << 16), __uint_as_float(t_0[11] & 4294901760u)));
                        acc_0[22] = z_1_27.x;
                        acc_0[23] = z_1_27.y;
                        float2 ss2_1_27 = fma_f32x2_rn_noftz(z_1_27, z_1_27, make_float2(ssx, ssy));
                        ssx = ss2_1_27.x;
                        ssy = ss2_1_27.y;
                        float2 a2_1_28 = make_float2(acc_0[24], acc_0[25]);
                        float2 z_1_28 = add_f32x2_noftz(a2_1_28, make_float2(__uint_as_float(t_0[12] << 16), __uint_as_float(t_0[12] & 4294901760u)));
                        acc_0[24] = z_1_28.x;
                        acc_0[25] = z_1_28.y;
                        float2 ss2_1_28 = fma_f32x2_rn_noftz(z_1_28, z_1_28, make_float2(ssx, ssy));
                        ssx = ss2_1_28.x;
                        ssy = ss2_1_28.y;
                        float2 a2_1_29 = make_float2(acc_0[26], acc_0[27]);
                        float2 z_1_29 = add_f32x2_noftz(a2_1_29, make_float2(__uint_as_float(t_0[13] << 16), __uint_as_float(t_0[13] & 4294901760u)));
                        acc_0[26] = z_1_29.x;
                        acc_0[27] = z_1_29.y;
                        float2 ss2_1_29 = fma_f32x2_rn_noftz(z_1_29, z_1_29, make_float2(ssx, ssy));
                        ssx = ss2_1_29.x;
                        ssy = ss2_1_29.y;
                        float2 a2_1_30 = make_float2(acc_0[28], acc_0[29]);
                        float2 z_1_30 = add_f32x2_noftz(a2_1_30, make_float2(__uint_as_float(t_0[14] << 16), __uint_as_float(t_0[14] & 4294901760u)));
                        acc_0[28] = z_1_30.x;
                        acc_0[29] = z_1_30.y;
                        float2 ss2_1_30 = fma_f32x2_rn_noftz(z_1_30, z_1_30, make_float2(ssx, ssy));
                        ssx = ss2_1_30.x;
                        ssy = ss2_1_30.y;
                        float2 a2_1_31 = make_float2(acc_0[30], acc_0[31]);
                        float2 z_1_31 = add_f32x2_noftz(a2_1_31, make_float2(__uint_as_float(t_0[15] << 16), __uint_as_float(t_0[15] & 4294901760u)));
                        acc_0[30] = z_1_31.x;
                        acc_0[31] = z_1_31.y;
                        float2 ss2_1_31 = fma_f32x2_rn_noftz(z_1_31, z_1_31, make_float2(ssx, ssy));
                        ssx = ss2_1_31.x;
                        ssy = ss2_1_31.y;
                        unsigned int acc_bf16_0[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
                            acc_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 0))), "r"(acc_bf16_0[0]), "r"(acc_bf16_0[1]), "r"(acc_bf16_0[2]), "r"(acc_bf16_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 1))), "r"(acc_bf16_0[4]), "r"(acc_bf16_0[5]), "r"(acc_bf16_0[6]), "r"(acc_bf16_0[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 2))), "r"(acc_bf16_0[8]), "r"(acc_bf16_0[9]), "r"(acc_bf16_0[10]), "r"(acc_bf16_0[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp1 + 16 * (f ^ 3))), "r"(acc_bf16_0[12]), "r"(acc_bf16_0[13]), "r"(acc_bf16_0[14]), "r"(acc_bf16_0[15]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_6 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(res_out) + (drow * 12288));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_6), "r"(wsm_addr + (unsigned int)rb_off), "r"((uint32_t)(12288))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    float ss;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ss) : "f"(ssx), "f"(ssy));
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                    float _add_7;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(ss), "f"(_shfl_xor_sync_8));
                    ss = _add_7;
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ss), "f"(_shfl_xor_sync_10));
                    ss = _add_9;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(ss), "f"(_shfl_xor_sync_12));
                    ss = _add_11;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ss), "f"(_shfl_xor_sync_14));
                    ss = _add_13;
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(ss), "f"(_shfl_xor_sync_16));
                    ss = _add_15;
                    float _fdiv_17 = __fdiv_rn(ss, 6144.0f);
                    float arg;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(arg) : "f"(_fdiv_17), "f"(eps));
                    float rinv;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(rinv) : "f"(arg));
                    float2 r2 = make_float2(rinv, rinv);
                    float2 b2 = make_float2(wbias, wbias);
                    #pragma unroll 24
                    for (int m = 0; m < 24; m++) {
                        int c = lane_2 + 32 * m;
                        unsigned int x4[4];
                        unsigned int w4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&x4[0])), "=r"(*reinterpret_cast<uint32_t*>(&x4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&x4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&x4[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 16 * c)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w4[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(16 * c)));
                        float y[8];
                        float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(x4[0] << 16), __uint_as_float(x4[0] & 4294901760u));
                        float2 tz_0;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&r2));
                        float2 tw_0 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[0] << 16), __uint_as_float(w4[0] & 4294901760u)));
                        float2 t_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&tz_0), "l"(*(const unsigned long long*)&tw_0));
                        y[0] = t_1.x;
                        y[1] = t_1.y;
                        float2 _f32x2_mul_a_19 = make_float2(__uint_as_float(x4[1] << 16), __uint_as_float(x4[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&r2));
                        float2 tw_1 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[1] << 16), __uint_as_float(w4[1] & 4294901760u)));
                        float2 t_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_1));
                        y[2] = t_2.x;
                        y[3] = t_2.y;
                        float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(x4[2] << 16), __uint_as_float(x4[2] & 4294901760u));
                        float2 tz_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[2] << 16), __uint_as_float(w4[2] & 4294901760u)));
                        float2 t_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&tz_2), "l"(*(const unsigned long long*)&tw_2));
                        y[4] = t_3.x;
                        y[5] = t_3.y;
                        float2 _f32x2_mul_a_21 = make_float2(__uint_as_float(x4[3] << 16), __uint_as_float(x4[3] & 4294901760u));
                        float2 tz_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&r2));
                        float2 tw_3 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[3] << 16), __uint_as_float(w4[3] & 4294901760u)));
                        float2 t_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&tz_3), "l"(*(const unsigned long long*)&tw_3));
                        y[6] = t_4.x;
                        y[7] = t_4.y;
                        unsigned int y_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        unsigned int v4[4];
                        v4[0] = y_bf16[0];
                        v4[1] = y_bf16[1];
                        v4[2] = y_bf16[2];
                        v4[3] = y_bf16[3];
                        long long idx = drow * 3072 + (long long)(4 * c);
                        {
                            int4 _iv4 = make_int4(v4[0 + 0], v4[0 + 1], v4[0 + 2], v4[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(out) + idx) = _iv4;
                        }
                    }
                    if (n > kd + 1) {
                        __syncwarp();
                        if (lane_2 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                            long long nrow = (long long)(r0 + warp_1 + 8 * (kd + 1));
                            mbarrier_arrive_expect_tx(rb_full_addr + (warp_1) * 8, 12288);
                            cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)rb_off, res_in + (nrow * 6144), 12288, rb_full_addr + (warp_1) * 8);
                        }
                    }
                }
                _phase_xb_full ^= 1;
                _phase_rb_full ^= 1;
            }
            if (lane_2 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_1 = bid * 4;
                ctrl[base_1 + 2] = (unsigned int)rows;
                ctrl[base_1 + 3] = 1;
                ctrl[base_1] = (call + 1) % 3;
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 8) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("barrier.sync 0, 288;" ::: "memory");
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
                long long lo_3 = u * bid64_1 / g64_1;
                long long hi_3 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_3 + lane64;
                #pragma unroll 1
                for (long long g = g0; g < hi_3; g += 32) {
                    long long widx = v_base_1 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_3_1 = u_0 * bid64_1 / g64_1;
                long long hi_3_1 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_0 = lo_3_1 + lane64;
                #pragma unroll 1
                for (long long g_1 = g0_0; g_1 < hi_3_1; g_1 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_1 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx_1) = _iv4;
                    }
                }
                long long u_1 = pr64 * 16 / 16;
                long long lo_4 = u_1 * bid64_1 / g64_1;
                long long hi_4 = u_1 * (bid64_1 + 1) / g64_1;
                long long g0_1 = lo_4 + lane64;
                #pragma unroll 1
                for (long long g_2 = g0_1; g_2 < hi_4; g_2 += 32) {
                    long long widx_2 = m_base_1 + 16 * g_2 >> 2;
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
