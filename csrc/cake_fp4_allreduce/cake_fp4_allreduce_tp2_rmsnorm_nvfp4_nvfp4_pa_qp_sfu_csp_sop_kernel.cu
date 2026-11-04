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
#define SMEM_WSM_STAGE_BYTES 198016
#define SMEM_WSM_STRIDE 198016
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 198016
#define SMEM_WSM32_STRIDE 198016
#define SMEM_TOTAL 211456
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 1
#ifndef CAKE_FP4_ALLREDUCE_KERNEL
#define CAKE_FP4_ALLREDUCE_KERNEL cake_fp4_allreduce_tp2_rmsnorm_nvfp4_nvfp4_pa_qp_sfu_csp_sop
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

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
CAKE_FP4_ALLREDUCE_KERNEL(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ outq, uint8_t* __restrict__ out_sf, float* __restrict__ gs_out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752 + 12464), res_in + (first_row * 6144), 12288, rb_full_addr + (warp_1) * 8);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752), xq + (first_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
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
            int xb_off = warp_1 * 24752;
            int own_off = xb_off + 12288;
            int stg_off = xb_off + 10416;
            int rb_off = stg_off + 2048;
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
                    unsigned int pv[24];
                    unsigned int ps[6];
                    int pp_5_0 = lane_2;
                    uint32_t _sysv_ld_0[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_0[0]), "=r"(_sysv_ld_0[1]), "=r"(_sysv_ld_0[2]), "=r"(_sysv_ld_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_0)))) : "memory");
                    pv[0] = _sysv_ld_0[0];
                    pv[1] = _sysv_ld_0[1];
                    pv[2] = _sysv_ld_0[2];
                    pv[3] = _sysv_ld_0[3];
                    uint32_t _sysv_u32_0;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_0 >> 1))))) : "memory");
                    ps[0] = _sysv_u32_0 >> (unsigned int)(16 * (pp_5_0 & 1)) & 65535;
                    int pp_5_1 = lane_2 + 32;
                    uint32_t _sysv_ld_1[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_1[0]), "=r"(_sysv_ld_1[1]), "=r"(_sysv_ld_1[2]), "=r"(_sysv_ld_1[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_1)))) : "memory");
                    pv[4] = _sysv_ld_1[0];
                    pv[5] = _sysv_ld_1[1];
                    pv[6] = _sysv_ld_1[2];
                    pv[7] = _sysv_ld_1[3];
                    uint32_t _sysv_u32_1;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_1) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_1 >> 1))))) : "memory");
                    ps[1] = _sysv_u32_1 >> (unsigned int)(16 * (pp_5_1 & 1)) & 65535;
                    int pp_5_2 = lane_2 + 64;
                    uint32_t _sysv_ld_2[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]), "=r"(_sysv_ld_2[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_2)))) : "memory");
                    pv[8] = _sysv_ld_2[0];
                    pv[9] = _sysv_ld_2[1];
                    pv[10] = _sysv_ld_2[2];
                    pv[11] = _sysv_ld_2[3];
                    uint32_t _sysv_u32_2;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_2) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_2 >> 1))))) : "memory");
                    ps[2] = _sysv_u32_2 >> (unsigned int)(16 * (pp_5_2 & 1)) & 65535;
                    int pp_5_3 = lane_2 + 96;
                    uint32_t _sysv_ld_3[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_3[0]), "=r"(_sysv_ld_3[1]), "=r"(_sysv_ld_3[2]), "=r"(_sysv_ld_3[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_3)))) : "memory");
                    pv[12] = _sysv_ld_3[0];
                    pv[13] = _sysv_ld_3[1];
                    pv[14] = _sysv_ld_3[2];
                    pv[15] = _sysv_ld_3[3];
                    uint32_t _sysv_u32_3;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_3) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_3 >> 1))))) : "memory");
                    ps[3] = _sysv_u32_3 >> (unsigned int)(16 * (pp_5_3 & 1)) & 65535;
                    int pp_5_4 = lane_2 + 128;
                    uint32_t _sysv_ld_4[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_4[0]), "=r"(_sysv_ld_4[1]), "=r"(_sysv_ld_4[2]), "=r"(_sysv_ld_4[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_4)))) : "memory");
                    pv[16] = _sysv_ld_4[0];
                    pv[17] = _sysv_ld_4[1];
                    pv[18] = _sysv_ld_4[2];
                    pv[19] = _sysv_ld_4[3];
                    uint32_t _sysv_u32_4;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_4) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_4 >> 1))))) : "memory");
                    ps[4] = _sysv_u32_4 >> (unsigned int)(16 * (pp_5_4 & 1)) & 65535;
                    int pp_5_5 = lane_2 + 160;
                    uint32_t _sysv_ld_5[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_5[0]), "=r"(_sysv_ld_5[1]), "=r"(_sysv_ld_5[2]), "=r"(_sysv_ld_5[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_5_5)))) : "memory");
                    pv[20] = _sysv_ld_5[0];
                    pv[21] = _sysv_ld_5[1];
                    pv[22] = _sysv_ld_5[2];
                    pv[23] = _sysv_ld_5[3];
                    uint32_t _sysv_u32_5;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_5) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_5_5 >> 1))))) : "memory");
                    ps[5] = _sysv_u32_5 >> (unsigned int)(16 * (pp_5_5 & 1)) & 65535;
                    unsigned int rsw[1];
                    uint32_t word;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(word) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = word;
                    float ssx = 0.0f;
                    float ssy = 0.0f;
                    unsigned int d = 0;
                    d = d | pv[0] & ~((pv[0] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[1] & ~((pv[1] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[2] & ~((pv[2] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[3] & ~((pv[3] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | ((ps[0] & 255) + 1 | (ps[0] >> 8) + 1) & 256;
                    d = d | pv[4] & ~((pv[4] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[5] & ~((pv[5] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[6] & ~((pv[6] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | pv[7] & ~((pv[7] & 2004318071) + 2004318071) & 2290649224u;
                    d = d | ((ps[1] & 255) + 1 | (ps[1] >> 8) + 1) & 256;
                    if (d != 0) {
                        int pp0 = lane_2;
                        int pp1 = pp0 + 32;
                        uint32_t _fp4_wire_values_0[8];
                        uint32_t _fp4_wire_scales_0[2];
                        uint16_t _wire_h16_5_0;
                        uint16_t _wire_h16_5_1;
                        uint32_t _wire_vpend_5 = 3u;
                        uint32_t _wire_spend_5 = 3u;
                        do {
                            if (_wire_vpend_5 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[0]), "=r"(_fp4_wire_values_0[1]), "=r"(_fp4_wire_values_0[2]), "=r"(_fp4_wire_values_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0)))) : "memory");
                            }
                            if (_wire_vpend_5 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_0[4]), "=r"(_fp4_wire_values_0[5]), "=r"(_fp4_wire_values_0[6]), "=r"(_fp4_wire_values_0[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1)))) : "memory");
                            }
                            if (_wire_spend_5 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0)))) : "memory");
                                _fp4_wire_scales_0[0] = (uint32_t)_wire_h16_5_0;
                            }
                            if (_wire_spend_5 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_5_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1)))) : "memory");
                                _fp4_wire_scales_0[1] = (uint32_t)_wire_h16_5_1;
                            }
                            _wire_vpend_5 = ((uint32_t)((((_fp4_wire_values_0[0] & ~((_fp4_wire_values_0[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[1] & ~((_fp4_wire_values_0[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[2] & ~((_fp4_wire_values_0[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[3] & ~((_fp4_wire_values_0[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_0[4] & ~((_fp4_wire_values_0[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[5] & ~((_fp4_wire_values_0[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[6] & ~((_fp4_wire_values_0[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_0[7] & ~((_fp4_wire_values_0[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_5 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_0[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_0[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_5 | _wire_spend_5) != 0u);
                        pv[0] = _fp4_wire_values_0[0];
                        pv[1] = _fp4_wire_values_0[1];
                        pv[2] = _fp4_wire_values_0[2];
                        pv[3] = _fp4_wire_values_0[3];
                        ps[0] = _fp4_wire_scales_0[0];
                        pv[4] = _fp4_wire_values_0[4];
                        pv[5] = _fp4_wire_values_0[5];
                        pv[6] = _fp4_wire_values_0[6];
                        pv[7] = _fp4_wire_values_0[7];
                        ps[1] = _fp4_wire_scales_0[1];
                    }
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
                    mbarrier_wait(rb_full_addr + (warp_1) * 8, _phase_rb_full);
                    float rs_peer_f_0 = __uint_as_float(rsw[0]);
                    float rs_a_f_0 = ((first) ? own_rs : rs_peer_f_0);
                    float rs_b_f_0 = ((first) ? rs_peer_f_0 : own_rs);
                    int pp_0 = lane_2;
                    unsigned int own[4];
                    unsigned int own_sc[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_0)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_0)));
                    unsigned int a[4];
                    unsigned int b[4];
                    a[0] = ((first) ? own[0] : pv[0]);
                    b[0] = ((first) ? pv[0] : own[0]);
                    a[1] = ((first) ? own[1] : pv[1]);
                    b[1] = ((first) ? pv[1] : own[1]);
                    a[2] = ((first) ? own[2] : pv[2]);
                    b[2] = ((first) ? pv[2] : own[2]);
                    a[3] = ((first) ? own[3] : pv[3]);
                    b[3] = ((first) ? pv[3] : own[3]);
                    unsigned int a_sc = ((first) ? own_sc[0] : ps[0]);
                    unsigned int b_sc = ((first) ? ps[0] : own_sc[0]);
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
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(cvt_f32_f16), "f"(rs_a_f_0));
                    float s1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(cvt_f32_f16_1), "f"(rs_a_f_0));
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
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(cvt_f32_f16_0), "f"(rs_b_f_0));
                    float s1_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_0) : "f"(cvt_f32_f16_1_0), "f"(rs_b_f_0));
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
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 3))));
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
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 0))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 1))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 2))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_0 + 16 * (f ^ 3))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                    int pp_1 = lane_2 + 32;
                    unsigned int own_0[4];
                    unsigned int own_sc_0[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_1)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_0[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_1)));
                    unsigned int a_32[4];
                    unsigned int b_0[4];
                    a_32[0] = ((first) ? own_0[0] : pv[4]);
                    b_0[0] = ((first) ? pv[4] : own_0[0]);
                    a_32[1] = ((first) ? own_0[1] : pv[5]);
                    b_0[1] = ((first) ? pv[5] : own_0[1]);
                    a_32[2] = ((first) ? own_0[2] : pv[6]);
                    b_0[2] = ((first) ? pv[6] : own_0[2]);
                    a_32[3] = ((first) ? own_0[3] : pv[7]);
                    b_0[3] = ((first) ? pv[7] : own_0[3]);
                    unsigned int a_sc_0 = ((first) ? own_sc_0[0] : ps[1]);
                    unsigned int b_sc_0 = ((first) ? ps[1] : own_sc_0[0]);
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
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(cvt_f32_f16_2), "f"(rs_a_f_0));
                    float s1_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_1) : "f"(cvt_f32_f16_1_1), "f"(rs_a_f_0));
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
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(cvt_f32_f16_3), "f"(rs_b_f_0));
                    float s1_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_2) : "f"(cvt_f32_f16_1_2), "f"(rs_b_f_0));
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
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_0[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 3))));
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
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 0))), "r"(acc_bf16_0[0]), "r"(acc_bf16_0[1]), "r"(acc_bf16_0[2]), "r"(acc_bf16_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 1))), "r"(acc_bf16_0[4]), "r"(acc_bf16_0[5]), "r"(acc_bf16_0[6]), "r"(acc_bf16_0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 2))), "r"(acc_bf16_0[8]), "r"(acc_bf16_0[9]), "r"(acc_bf16_0[10]), "r"(acc_bf16_0[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_1 + 16 * (f ^ 3))), "r"(acc_bf16_0[12]), "r"(acc_bf16_0[13]), "r"(acc_bf16_0[14]), "r"(acc_bf16_0[15]) : "memory");
                    unsigned int d_0 = 0;
                    d_0 = d_0 | pv[8] & ~((pv[8] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[9] & ~((pv[9] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[10] & ~((pv[10] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[11] & ~((pv[11] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | ((ps[2] & 255) + 1 | (ps[2] >> 8) + 1) & 256;
                    d_0 = d_0 | pv[12] & ~((pv[12] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[13] & ~((pv[13] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[14] & ~((pv[14] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | pv[15] & ~((pv[15] & 2004318071) + 2004318071) & 2290649224u;
                    d_0 = d_0 | ((ps[3] & 255) + 1 | (ps[3] >> 8) + 1) & 256;
                    if (d_0 != 0) {
                        int pp0_1 = lane_2 + 64;
                        int pp1_1 = pp0_1 + 32;
                        uint32_t fp4_wire_values[8];
                        uint32_t fp4_wire_scales[2];
                        uint16_t _wire_h16_6_0;
                        uint16_t _wire_h16_6_1;
                        uint32_t _wire_vpend_6 = 3u;
                        uint32_t _wire_spend_6 = 3u;
                        do {
                            if (_wire_vpend_6 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[0]), "=r"(fp4_wire_values[1]), "=r"(fp4_wire_values[2]), "=r"(fp4_wire_values[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0_1)))) : "memory");
                            }
                            if (_wire_vpend_6 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values[4]), "=r"(fp4_wire_values[5]), "=r"(fp4_wire_values[6]), "=r"(fp4_wire_values[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1_1)))) : "memory");
                            }
                            if (_wire_spend_6 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_6_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0_1)))) : "memory");
                                fp4_wire_scales[0] = (uint32_t)_wire_h16_6_0;
                            }
                            if (_wire_spend_6 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_6_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1_1)))) : "memory");
                                fp4_wire_scales[1] = (uint32_t)_wire_h16_6_1;
                            }
                            _wire_vpend_6 = ((uint32_t)((((fp4_wire_values[0] & ~((fp4_wire_values[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[1] & ~((fp4_wire_values[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[2] & ~((fp4_wire_values[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[3] & ~((fp4_wire_values[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values[4] & ~((fp4_wire_values[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[5] & ~((fp4_wire_values[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[6] & ~((fp4_wire_values[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values[7] & ~((fp4_wire_values[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_6 = ((uint32_t)(((uint32_t)((fp4_wire_scales[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_6 | _wire_spend_6) != 0u);
                        pv[8] = fp4_wire_values[0];
                        pv[9] = fp4_wire_values[1];
                        pv[10] = fp4_wire_values[2];
                        pv[11] = fp4_wire_values[3];
                        ps[2] = fp4_wire_scales[0];
                        pv[12] = fp4_wire_values[4];
                        pv[13] = fp4_wire_values[5];
                        pv[14] = fp4_wire_values[6];
                        pv[15] = fp4_wire_values[7];
                        ps[3] = fp4_wire_scales[1];
                    }
                    int pp_2 = lane_2 + 64;
                    unsigned int own_1[4];
                    unsigned int own_sc_1[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_2)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_1[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_2)));
                    unsigned int a_65[4];
                    unsigned int b_1[4];
                    a_65[0] = ((first) ? own_1[0] : pv[8]);
                    b_1[0] = ((first) ? pv[8] : own_1[0]);
                    a_65[1] = ((first) ? own_1[1] : pv[9]);
                    b_1[1] = ((first) ? pv[9] : own_1[1]);
                    a_65[2] = ((first) ? own_1[2] : pv[10]);
                    b_1[2] = ((first) ? pv[10] : own_1[2]);
                    a_65[3] = ((first) ? own_1[3] : pv[11]);
                    b_1[3] = ((first) ? pv[11] : own_1[3]);
                    unsigned int a_sc_1 = ((first) ? own_sc_1[0] : ps[2]);
                    unsigned int b_sc_1 = ((first) ? ps[2] : own_sc_1[0]);
                    float acc_1[32] = {0};
                    bool fhi_5 = (f & 2) != 0;
                    bool fodd_3 = (f & 1) != 0;
                    bool fhi_6 = (f & 2) != 0;
                    unsigned int s_3[4];
                    s_3[0] = a_65[0];
                    s_3[1] = a_65[1];
                    s_3[2] = a_65[2];
                    s_3[3] = a_65[3];
                    if (fodd_3) {
                        s_3[0] = a_65[1];
                        s_3[1] = a_65[0];
                        s_3[2] = a_65[3];
                        s_3[3] = a_65[2];
                    }
                    unsigned int r_67[4];
                    r_67[0] = s_3[0];
                    r_67[1] = s_3[1];
                    r_67[2] = s_3[2];
                    r_67[3] = s_3[3];
                    if (fhi_6) {
                        r_67[0] = s_3[2];
                        r_67[1] = s_3[3];
                        r_67[2] = s_3[0];
                        r_67[3] = s_3[1];
                    }
                    bool fodd_4 = (f & 1) != 0;
                    bool fhi_7 = (f & 2) != 0;
                    unsigned int s_4[4];
                    s_4[0] = b_1[0];
                    s_4[1] = b_1[1];
                    s_4[2] = b_1[2];
                    s_4[3] = b_1[3];
                    if (fodd_4) {
                        s_4[0] = b_1[1];
                        s_4[1] = b_1[0];
                        s_4[2] = b_1[3];
                        s_4[3] = b_1[2];
                    }
                    unsigned int r_68[4];
                    r_68[0] = s_4[0];
                    r_68[1] = s_4[1];
                    r_68[2] = s_4[2];
                    r_68[3] = s_4[3];
                    if (fhi_7) {
                        r_68[0] = s_4[2];
                        r_68[1] = s_4[3];
                        r_68[2] = s_4[0];
                        r_68[3] = s_4[1];
                    }
                    uint32_t e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(a_sc_1)));
                    float cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                    float cvt_f32_f16_1_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                    float s0_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_3) : "f"(cvt_f32_f16_4), "f"(rs_a_f_0));
                    float s1_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_3) : "f"(cvt_f32_f16_1_3), "f"(rs_a_f_0));
                    float lo_3 = s0_3;
                    float hi_3 = s1_3;
                    if (fhi_5) {
                        lo_3 = s1_3;
                        hi_3 = s0_3;
                    }
                    float2 scale0_3 = make_float2(lo_3, lo_3);
                    float2 scale1_3 = make_float2(hi_3, hi_3);
                    float tv_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_67[0]));
                        (&tv_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_16[0]), "=f"(tv_16[1]), "=f"(tv_16[2]), "=f"(tv_16[3]), "=f"(tv_16[4]), "=f"(tv_16[5]), "=f"(tv_16[6]), "=f"(tv_16[7]) : "r"(r_67[0]));
                        #endif
                    }
                    float2 a_66 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_64 = make_float2(tv_16[0], tv_16[1]);
                    float2 r_69 = fma_f32x2_rn_noftz(q_64, scale0_3, a_66);
                    acc_1[0] = r_69.x;
                    acc_1[1] = r_69.y;
                    float2 a_67 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_65 = make_float2(tv_16[2], tv_16[3]);
                    float2 r_70 = fma_f32x2_rn_noftz(q_65, scale0_3, a_67);
                    acc_1[2] = r_70.x;
                    acc_1[3] = r_70.y;
                    float2 a_68 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_66 = make_float2(tv_16[4], tv_16[5]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_66, scale0_3, a_68);
                    acc_1[4] = r_71.x;
                    acc_1[5] = r_71.y;
                    float2 a_69 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_67 = make_float2(tv_16[6], tv_16[7]);
                    float2 r_72 = fma_f32x2_rn_noftz(q_67, scale0_3, a_69);
                    acc_1[6] = r_72.x;
                    acc_1[7] = r_72.y;
                    float tv_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_67[1 + 0]));
                        (&tv_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_17[0]), "=f"(tv_17[1]), "=f"(tv_17[2]), "=f"(tv_17[3]), "=f"(tv_17[4]), "=f"(tv_17[5]), "=f"(tv_17[6]), "=f"(tv_17[7]) : "r"(r_67[1 + 0]));
                        #endif
                    }
                    float2 a_70 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_68 = make_float2(tv_17[0], tv_17[1]);
                    float2 r_73 = fma_f32x2_rn_noftz(q_68, scale0_3, a_70);
                    acc_1[8] = r_73.x;
                    acc_1[9] = r_73.y;
                    float2 a_71 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_69 = make_float2(tv_17[2], tv_17[3]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_69, scale0_3, a_71);
                    acc_1[10] = r_74.x;
                    acc_1[11] = r_74.y;
                    float2 a_72 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_70 = make_float2(tv_17[4], tv_17[5]);
                    float2 r_75 = fma_f32x2_rn_noftz(q_70, scale0_3, a_72);
                    acc_1[12] = r_75.x;
                    acc_1[13] = r_75.y;
                    float2 a_73 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_71 = make_float2(tv_17[6], tv_17[7]);
                    float2 r_76 = fma_f32x2_rn_noftz(q_71, scale0_3, a_73);
                    acc_1[14] = r_76.x;
                    acc_1[15] = r_76.y;
                    float tv_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_67[2 + 0]));
                        (&tv_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_18[0]), "=f"(tv_18[1]), "=f"(tv_18[2]), "=f"(tv_18[3]), "=f"(tv_18[4]), "=f"(tv_18[5]), "=f"(tv_18[6]), "=f"(tv_18[7]) : "r"(r_67[2 + 0]));
                        #endif
                    }
                    float2 a_74 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_72 = make_float2(tv_18[0], tv_18[1]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_72, scale1_3, a_74);
                    acc_1[16] = r_77.x;
                    acc_1[17] = r_77.y;
                    float2 a_75 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_73 = make_float2(tv_18[2], tv_18[3]);
                    float2 r_78 = fma_f32x2_rn_noftz(q_73, scale1_3, a_75);
                    acc_1[18] = r_78.x;
                    acc_1[19] = r_78.y;
                    float2 a_76 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_74 = make_float2(tv_18[4], tv_18[5]);
                    float2 r_79 = fma_f32x2_rn_noftz(q_74, scale1_3, a_76);
                    acc_1[20] = r_79.x;
                    acc_1[21] = r_79.y;
                    float2 a_77 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_75 = make_float2(tv_18[6], tv_18[7]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_75, scale1_3, a_77);
                    acc_1[22] = r_80.x;
                    acc_1[23] = r_80.y;
                    float tv_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_67[3 + 0]));
                        (&tv_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_19[0]), "=f"(tv_19[1]), "=f"(tv_19[2]), "=f"(tv_19[3]), "=f"(tv_19[4]), "=f"(tv_19[5]), "=f"(tv_19[6]), "=f"(tv_19[7]) : "r"(r_67[3 + 0]));
                        #endif
                    }
                    float2 a_78 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_76 = make_float2(tv_19[0], tv_19[1]);
                    float2 r_81 = fma_f32x2_rn_noftz(q_76, scale1_3, a_78);
                    acc_1[24] = r_81.x;
                    acc_1[25] = r_81.y;
                    float2 a_79 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_77 = make_float2(tv_19[2], tv_19[3]);
                    float2 r_82 = fma_f32x2_rn_noftz(q_77, scale1_3, a_79);
                    acc_1[26] = r_82.x;
                    acc_1[27] = r_82.y;
                    float2 a_80 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_78 = make_float2(tv_19[4], tv_19[5]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_78, scale1_3, a_80);
                    acc_1[28] = r_83.x;
                    acc_1[29] = r_83.y;
                    float2 a_81 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_79 = make_float2(tv_19[6], tv_19[7]);
                    float2 r_84 = fma_f32x2_rn_noftz(q_79, scale1_3, a_81);
                    acc_1[30] = r_84.x;
                    acc_1[31] = r_84.y;
                    uint32_t e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(b_sc_1)));
                    float cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                    float cvt_f32_f16_1_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                    float s0_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_4) : "f"(cvt_f32_f16_5), "f"(rs_b_f_0));
                    float s1_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_4) : "f"(cvt_f32_f16_1_4), "f"(rs_b_f_0));
                    float lo_4 = s0_4;
                    float hi_4 = s1_4;
                    if (fhi_5) {
                        lo_4 = s1_4;
                        hi_4 = s0_4;
                    }
                    float2 scale0_4 = make_float2(lo_4, lo_4);
                    float2 scale1_4 = make_float2(hi_4, hi_4);
                    float tv_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_68[0]));
                        (&tv_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_20[0]), "=f"(tv_20[1]), "=f"(tv_20[2]), "=f"(tv_20[3]), "=f"(tv_20[4]), "=f"(tv_20[5]), "=f"(tv_20[6]), "=f"(tv_20[7]) : "r"(r_68[0]));
                        #endif
                    }
                    float2 a_82 = make_float2(acc_1[0], acc_1[1]);
                    float2 q_80 = make_float2(tv_20[0], tv_20[1]);
                    float2 r_85 = fma_f32x2_rn_noftz(q_80, scale0_4, a_82);
                    acc_1[0] = r_85.x;
                    acc_1[1] = r_85.y;
                    float2 a_83 = make_float2(acc_1[2], acc_1[3]);
                    float2 q_81 = make_float2(tv_20[2], tv_20[3]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_81, scale0_4, a_83);
                    acc_1[2] = r_86.x;
                    acc_1[3] = r_86.y;
                    float2 a_84 = make_float2(acc_1[4], acc_1[5]);
                    float2 q_82 = make_float2(tv_20[4], tv_20[5]);
                    float2 r_87 = fma_f32x2_rn_noftz(q_82, scale0_4, a_84);
                    acc_1[4] = r_87.x;
                    acc_1[5] = r_87.y;
                    float2 a_85 = make_float2(acc_1[6], acc_1[7]);
                    float2 q_83 = make_float2(tv_20[6], tv_20[7]);
                    float2 r_88 = fma_f32x2_rn_noftz(q_83, scale0_4, a_85);
                    acc_1[6] = r_88.x;
                    acc_1[7] = r_88.y;
                    float tv_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_68[1 + 0]));
                        (&tv_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_21[0]), "=f"(tv_21[1]), "=f"(tv_21[2]), "=f"(tv_21[3]), "=f"(tv_21[4]), "=f"(tv_21[5]), "=f"(tv_21[6]), "=f"(tv_21[7]) : "r"(r_68[1 + 0]));
                        #endif
                    }
                    float2 a_86 = make_float2(acc_1[8], acc_1[9]);
                    float2 q_84 = make_float2(tv_21[0], tv_21[1]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_84, scale0_4, a_86);
                    acc_1[8] = r_89.x;
                    acc_1[9] = r_89.y;
                    float2 a_87 = make_float2(acc_1[10], acc_1[11]);
                    float2 q_85 = make_float2(tv_21[2], tv_21[3]);
                    float2 r_90 = fma_f32x2_rn_noftz(q_85, scale0_4, a_87);
                    acc_1[10] = r_90.x;
                    acc_1[11] = r_90.y;
                    float2 a_88 = make_float2(acc_1[12], acc_1[13]);
                    float2 q_86 = make_float2(tv_21[4], tv_21[5]);
                    float2 r_91 = fma_f32x2_rn_noftz(q_86, scale0_4, a_88);
                    acc_1[12] = r_91.x;
                    acc_1[13] = r_91.y;
                    float2 a_89 = make_float2(acc_1[14], acc_1[15]);
                    float2 q_87 = make_float2(tv_21[6], tv_21[7]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_87, scale0_4, a_89);
                    acc_1[14] = r_92.x;
                    acc_1[15] = r_92.y;
                    float tv_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_68[2 + 0]));
                        (&tv_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_22[0]), "=f"(tv_22[1]), "=f"(tv_22[2]), "=f"(tv_22[3]), "=f"(tv_22[4]), "=f"(tv_22[5]), "=f"(tv_22[6]), "=f"(tv_22[7]) : "r"(r_68[2 + 0]));
                        #endif
                    }
                    float2 a_90 = make_float2(acc_1[16], acc_1[17]);
                    float2 q_88 = make_float2(tv_22[0], tv_22[1]);
                    float2 r_93 = fma_f32x2_rn_noftz(q_88, scale1_4, a_90);
                    acc_1[16] = r_93.x;
                    acc_1[17] = r_93.y;
                    float2 a_91 = make_float2(acc_1[18], acc_1[19]);
                    float2 q_89 = make_float2(tv_22[2], tv_22[3]);
                    float2 r_94 = fma_f32x2_rn_noftz(q_89, scale1_4, a_91);
                    acc_1[18] = r_94.x;
                    acc_1[19] = r_94.y;
                    float2 a_92 = make_float2(acc_1[20], acc_1[21]);
                    float2 q_90 = make_float2(tv_22[4], tv_22[5]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_90, scale1_4, a_92);
                    acc_1[20] = r_95.x;
                    acc_1[21] = r_95.y;
                    float2 a_93 = make_float2(acc_1[22], acc_1[23]);
                    float2 q_91 = make_float2(tv_22[6], tv_22[7]);
                    float2 r_96 = fma_f32x2_rn_noftz(q_91, scale1_4, a_93);
                    acc_1[22] = r_96.x;
                    acc_1[23] = r_96.y;
                    float tv_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_68[3 + 0]));
                        (&tv_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_23[0]), "=f"(tv_23[1]), "=f"(tv_23[2]), "=f"(tv_23[3]), "=f"(tv_23[4]), "=f"(tv_23[5]), "=f"(tv_23[6]), "=f"(tv_23[7]) : "r"(r_68[3 + 0]));
                        #endif
                    }
                    float2 a_94 = make_float2(acc_1[24], acc_1[25]);
                    float2 q_92 = make_float2(tv_23[0], tv_23[1]);
                    float2 r_97 = fma_f32x2_rn_noftz(q_92, scale1_4, a_94);
                    acc_1[24] = r_97.x;
                    acc_1[25] = r_97.y;
                    float2 a_95 = make_float2(acc_1[26], acc_1[27]);
                    float2 q_93 = make_float2(tv_23[2], tv_23[3]);
                    float2 r_98 = fma_f32x2_rn_noftz(q_93, scale1_4, a_95);
                    acc_1[26] = r_98.x;
                    acc_1[27] = r_98.y;
                    float2 a_96 = make_float2(acc_1[28], acc_1[29]);
                    float2 q_94 = make_float2(tv_23[4], tv_23[5]);
                    float2 r_99 = fma_f32x2_rn_noftz(q_94, scale1_4, a_96);
                    acc_1[28] = r_99.x;
                    acc_1[29] = r_99.y;
                    float2 a_97 = make_float2(acc_1[30], acc_1[31]);
                    float2 q_95 = make_float2(tv_23[6], tv_23[7]);
                    float2 r_100 = fma_f32x2_rn_noftz(q_95, scale1_4, a_97);
                    acc_1[30] = r_100.x;
                    acc_1[31] = r_100.y;
                    unsigned int t_1[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_1[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 3))));
                    float2 a2_1_32 = make_float2(acc_1[0], acc_1[1]);
                    float2 z_1_32 = add_f32x2_noftz(a2_1_32, make_float2(__uint_as_float(t_1[0] << 16), __uint_as_float(t_1[0] & 4294901760u)));
                    acc_1[0] = z_1_32.x;
                    acc_1[1] = z_1_32.y;
                    float2 ss2_1_32 = fma_f32x2_rn_noftz(z_1_32, z_1_32, make_float2(ssx, ssy));
                    ssx = ss2_1_32.x;
                    ssy = ss2_1_32.y;
                    float2 a2_1_33 = make_float2(acc_1[2], acc_1[3]);
                    float2 z_1_33 = add_f32x2_noftz(a2_1_33, make_float2(__uint_as_float(t_1[1] << 16), __uint_as_float(t_1[1] & 4294901760u)));
                    acc_1[2] = z_1_33.x;
                    acc_1[3] = z_1_33.y;
                    float2 ss2_1_33 = fma_f32x2_rn_noftz(z_1_33, z_1_33, make_float2(ssx, ssy));
                    ssx = ss2_1_33.x;
                    ssy = ss2_1_33.y;
                    float2 a2_1_34 = make_float2(acc_1[4], acc_1[5]);
                    float2 z_1_34 = add_f32x2_noftz(a2_1_34, make_float2(__uint_as_float(t_1[2] << 16), __uint_as_float(t_1[2] & 4294901760u)));
                    acc_1[4] = z_1_34.x;
                    acc_1[5] = z_1_34.y;
                    float2 ss2_1_34 = fma_f32x2_rn_noftz(z_1_34, z_1_34, make_float2(ssx, ssy));
                    ssx = ss2_1_34.x;
                    ssy = ss2_1_34.y;
                    float2 a2_1_35 = make_float2(acc_1[6], acc_1[7]);
                    float2 z_1_35 = add_f32x2_noftz(a2_1_35, make_float2(__uint_as_float(t_1[3] << 16), __uint_as_float(t_1[3] & 4294901760u)));
                    acc_1[6] = z_1_35.x;
                    acc_1[7] = z_1_35.y;
                    float2 ss2_1_35 = fma_f32x2_rn_noftz(z_1_35, z_1_35, make_float2(ssx, ssy));
                    ssx = ss2_1_35.x;
                    ssy = ss2_1_35.y;
                    float2 a2_1_36 = make_float2(acc_1[8], acc_1[9]);
                    float2 z_1_36 = add_f32x2_noftz(a2_1_36, make_float2(__uint_as_float(t_1[4] << 16), __uint_as_float(t_1[4] & 4294901760u)));
                    acc_1[8] = z_1_36.x;
                    acc_1[9] = z_1_36.y;
                    float2 ss2_1_36 = fma_f32x2_rn_noftz(z_1_36, z_1_36, make_float2(ssx, ssy));
                    ssx = ss2_1_36.x;
                    ssy = ss2_1_36.y;
                    float2 a2_1_37 = make_float2(acc_1[10], acc_1[11]);
                    float2 z_1_37 = add_f32x2_noftz(a2_1_37, make_float2(__uint_as_float(t_1[5] << 16), __uint_as_float(t_1[5] & 4294901760u)));
                    acc_1[10] = z_1_37.x;
                    acc_1[11] = z_1_37.y;
                    float2 ss2_1_37 = fma_f32x2_rn_noftz(z_1_37, z_1_37, make_float2(ssx, ssy));
                    ssx = ss2_1_37.x;
                    ssy = ss2_1_37.y;
                    float2 a2_1_38 = make_float2(acc_1[12], acc_1[13]);
                    float2 z_1_38 = add_f32x2_noftz(a2_1_38, make_float2(__uint_as_float(t_1[6] << 16), __uint_as_float(t_1[6] & 4294901760u)));
                    acc_1[12] = z_1_38.x;
                    acc_1[13] = z_1_38.y;
                    float2 ss2_1_38 = fma_f32x2_rn_noftz(z_1_38, z_1_38, make_float2(ssx, ssy));
                    ssx = ss2_1_38.x;
                    ssy = ss2_1_38.y;
                    float2 a2_1_39 = make_float2(acc_1[14], acc_1[15]);
                    float2 z_1_39 = add_f32x2_noftz(a2_1_39, make_float2(__uint_as_float(t_1[7] << 16), __uint_as_float(t_1[7] & 4294901760u)));
                    acc_1[14] = z_1_39.x;
                    acc_1[15] = z_1_39.y;
                    float2 ss2_1_39 = fma_f32x2_rn_noftz(z_1_39, z_1_39, make_float2(ssx, ssy));
                    ssx = ss2_1_39.x;
                    ssy = ss2_1_39.y;
                    float2 a2_1_40 = make_float2(acc_1[16], acc_1[17]);
                    float2 z_1_40 = add_f32x2_noftz(a2_1_40, make_float2(__uint_as_float(t_1[8] << 16), __uint_as_float(t_1[8] & 4294901760u)));
                    acc_1[16] = z_1_40.x;
                    acc_1[17] = z_1_40.y;
                    float2 ss2_1_40 = fma_f32x2_rn_noftz(z_1_40, z_1_40, make_float2(ssx, ssy));
                    ssx = ss2_1_40.x;
                    ssy = ss2_1_40.y;
                    float2 a2_1_41 = make_float2(acc_1[18], acc_1[19]);
                    float2 z_1_41 = add_f32x2_noftz(a2_1_41, make_float2(__uint_as_float(t_1[9] << 16), __uint_as_float(t_1[9] & 4294901760u)));
                    acc_1[18] = z_1_41.x;
                    acc_1[19] = z_1_41.y;
                    float2 ss2_1_41 = fma_f32x2_rn_noftz(z_1_41, z_1_41, make_float2(ssx, ssy));
                    ssx = ss2_1_41.x;
                    ssy = ss2_1_41.y;
                    float2 a2_1_42 = make_float2(acc_1[20], acc_1[21]);
                    float2 z_1_42 = add_f32x2_noftz(a2_1_42, make_float2(__uint_as_float(t_1[10] << 16), __uint_as_float(t_1[10] & 4294901760u)));
                    acc_1[20] = z_1_42.x;
                    acc_1[21] = z_1_42.y;
                    float2 ss2_1_42 = fma_f32x2_rn_noftz(z_1_42, z_1_42, make_float2(ssx, ssy));
                    ssx = ss2_1_42.x;
                    ssy = ss2_1_42.y;
                    float2 a2_1_43 = make_float2(acc_1[22], acc_1[23]);
                    float2 z_1_43 = add_f32x2_noftz(a2_1_43, make_float2(__uint_as_float(t_1[11] << 16), __uint_as_float(t_1[11] & 4294901760u)));
                    acc_1[22] = z_1_43.x;
                    acc_1[23] = z_1_43.y;
                    float2 ss2_1_43 = fma_f32x2_rn_noftz(z_1_43, z_1_43, make_float2(ssx, ssy));
                    ssx = ss2_1_43.x;
                    ssy = ss2_1_43.y;
                    float2 a2_1_44 = make_float2(acc_1[24], acc_1[25]);
                    float2 z_1_44 = add_f32x2_noftz(a2_1_44, make_float2(__uint_as_float(t_1[12] << 16), __uint_as_float(t_1[12] & 4294901760u)));
                    acc_1[24] = z_1_44.x;
                    acc_1[25] = z_1_44.y;
                    float2 ss2_1_44 = fma_f32x2_rn_noftz(z_1_44, z_1_44, make_float2(ssx, ssy));
                    ssx = ss2_1_44.x;
                    ssy = ss2_1_44.y;
                    float2 a2_1_45 = make_float2(acc_1[26], acc_1[27]);
                    float2 z_1_45 = add_f32x2_noftz(a2_1_45, make_float2(__uint_as_float(t_1[13] << 16), __uint_as_float(t_1[13] & 4294901760u)));
                    acc_1[26] = z_1_45.x;
                    acc_1[27] = z_1_45.y;
                    float2 ss2_1_45 = fma_f32x2_rn_noftz(z_1_45, z_1_45, make_float2(ssx, ssy));
                    ssx = ss2_1_45.x;
                    ssy = ss2_1_45.y;
                    float2 a2_1_46 = make_float2(acc_1[28], acc_1[29]);
                    float2 z_1_46 = add_f32x2_noftz(a2_1_46, make_float2(__uint_as_float(t_1[14] << 16), __uint_as_float(t_1[14] & 4294901760u)));
                    acc_1[28] = z_1_46.x;
                    acc_1[29] = z_1_46.y;
                    float2 ss2_1_46 = fma_f32x2_rn_noftz(z_1_46, z_1_46, make_float2(ssx, ssy));
                    ssx = ss2_1_46.x;
                    ssy = ss2_1_46.y;
                    float2 a2_1_47 = make_float2(acc_1[30], acc_1[31]);
                    float2 z_1_47 = add_f32x2_noftz(a2_1_47, make_float2(__uint_as_float(t_1[15] << 16), __uint_as_float(t_1[15] & 4294901760u)));
                    acc_1[30] = z_1_47.x;
                    acc_1[31] = z_1_47.y;
                    float2 ss2_1_47 = fma_f32x2_rn_noftz(z_1_47, z_1_47, make_float2(ssx, ssy));
                    ssx = ss2_1_47.x;
                    ssy = ss2_1_47.y;
                    unsigned int acc_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                        acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 0))), "r"(acc_bf16_1[0]), "r"(acc_bf16_1[1]), "r"(acc_bf16_1[2]), "r"(acc_bf16_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 1))), "r"(acc_bf16_1[4]), "r"(acc_bf16_1[5]), "r"(acc_bf16_1[6]), "r"(acc_bf16_1[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 2))), "r"(acc_bf16_1[8]), "r"(acc_bf16_1[9]), "r"(acc_bf16_1[10]), "r"(acc_bf16_1[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_2 + 16 * (f ^ 3))), "r"(acc_bf16_1[12]), "r"(acc_bf16_1[13]), "r"(acc_bf16_1[14]), "r"(acc_bf16_1[15]) : "memory");
                    int pp_3 = lane_2 + 96;
                    unsigned int own_2[4];
                    unsigned int own_sc_2[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_3)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_2[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_3)));
                    unsigned int a_98[4];
                    unsigned int b_2[4];
                    a_98[0] = ((first) ? own_2[0] : pv[12]);
                    b_2[0] = ((first) ? pv[12] : own_2[0]);
                    a_98[1] = ((first) ? own_2[1] : pv[13]);
                    b_2[1] = ((first) ? pv[13] : own_2[1]);
                    a_98[2] = ((first) ? own_2[2] : pv[14]);
                    b_2[2] = ((first) ? pv[14] : own_2[2]);
                    a_98[3] = ((first) ? own_2[3] : pv[15]);
                    b_2[3] = ((first) ? pv[15] : own_2[3]);
                    unsigned int a_sc_2 = ((first) ? own_sc_2[0] : ps[3]);
                    unsigned int b_sc_2 = ((first) ? ps[3] : own_sc_2[0]);
                    float acc_2[32] = {0};
                    bool fhi_8 = (f & 2) != 0;
                    bool fodd_5 = (f & 1) != 0;
                    bool fhi_9 = (f & 2) != 0;
                    unsigned int s_5[4];
                    s_5[0] = a_98[0];
                    s_5[1] = a_98[1];
                    s_5[2] = a_98[2];
                    s_5[3] = a_98[3];
                    if (fodd_5) {
                        s_5[0] = a_98[1];
                        s_5[1] = a_98[0];
                        s_5[2] = a_98[3];
                        s_5[3] = a_98[2];
                    }
                    unsigned int r_101[4];
                    r_101[0] = s_5[0];
                    r_101[1] = s_5[1];
                    r_101[2] = s_5[2];
                    r_101[3] = s_5[3];
                    if (fhi_9) {
                        r_101[0] = s_5[2];
                        r_101[1] = s_5[3];
                        r_101[2] = s_5[0];
                        r_101[3] = s_5[1];
                    }
                    bool fodd_6 = (f & 1) != 0;
                    bool fhi_10 = (f & 2) != 0;
                    unsigned int s_6[4];
                    s_6[0] = b_2[0];
                    s_6[1] = b_2[1];
                    s_6[2] = b_2[2];
                    s_6[3] = b_2[3];
                    if (fodd_6) {
                        s_6[0] = b_2[1];
                        s_6[1] = b_2[0];
                        s_6[2] = b_2[3];
                        s_6[3] = b_2[2];
                    }
                    unsigned int r_102[4];
                    r_102[0] = s_6[0];
                    r_102[1] = s_6[1];
                    r_102[2] = s_6[2];
                    r_102[3] = s_6[3];
                    if (fhi_10) {
                        r_102[0] = s_6[2];
                        r_102[1] = s_6[3];
                        r_102[2] = s_6[0];
                        r_102[3] = s_6[1];
                    }
                    uint32_t e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5) : "h"((uint16_t)(a_sc_2)));
                    float cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6) : "h"((uint16_t)(e4m3x2_to_f16x2_5)));
                    float cvt_f32_f16_1_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5) : "h"((uint16_t)(e4m3x2_to_f16x2_5 >> 16)));
                    float s0_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_5) : "f"(cvt_f32_f16_6), "f"(rs_a_f_0));
                    float s1_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_5) : "f"(cvt_f32_f16_1_5), "f"(rs_a_f_0));
                    float lo_5 = s0_5;
                    float hi_5 = s1_5;
                    if (fhi_8) {
                        lo_5 = s1_5;
                        hi_5 = s0_5;
                    }
                    float2 scale0_5 = make_float2(lo_5, lo_5);
                    float2 scale1_5 = make_float2(hi_5, hi_5);
                    float tv_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_101[0]));
                        (&tv_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_24[0]), "=f"(tv_24[1]), "=f"(tv_24[2]), "=f"(tv_24[3]), "=f"(tv_24[4]), "=f"(tv_24[5]), "=f"(tv_24[6]), "=f"(tv_24[7]) : "r"(r_101[0]));
                        #endif
                    }
                    float2 a_99 = make_float2(acc_2[0], acc_2[1]);
                    float2 q_96 = make_float2(tv_24[0], tv_24[1]);
                    float2 r_103 = fma_f32x2_rn_noftz(q_96, scale0_5, a_99);
                    acc_2[0] = r_103.x;
                    acc_2[1] = r_103.y;
                    float2 a_100 = make_float2(acc_2[2], acc_2[3]);
                    float2 q_97 = make_float2(tv_24[2], tv_24[3]);
                    float2 r_104 = fma_f32x2_rn_noftz(q_97, scale0_5, a_100);
                    acc_2[2] = r_104.x;
                    acc_2[3] = r_104.y;
                    float2 a_101 = make_float2(acc_2[4], acc_2[5]);
                    float2 q_98 = make_float2(tv_24[4], tv_24[5]);
                    float2 r_105 = fma_f32x2_rn_noftz(q_98, scale0_5, a_101);
                    acc_2[4] = r_105.x;
                    acc_2[5] = r_105.y;
                    float2 a_102 = make_float2(acc_2[6], acc_2[7]);
                    float2 q_99 = make_float2(tv_24[6], tv_24[7]);
                    float2 r_106 = fma_f32x2_rn_noftz(q_99, scale0_5, a_102);
                    acc_2[6] = r_106.x;
                    acc_2[7] = r_106.y;
                    float tv_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_101[1 + 0]));
                        (&tv_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_25[0]), "=f"(tv_25[1]), "=f"(tv_25[2]), "=f"(tv_25[3]), "=f"(tv_25[4]), "=f"(tv_25[5]), "=f"(tv_25[6]), "=f"(tv_25[7]) : "r"(r_101[1 + 0]));
                        #endif
                    }
                    float2 a_103 = make_float2(acc_2[8], acc_2[9]);
                    float2 q_100 = make_float2(tv_25[0], tv_25[1]);
                    float2 r_107 = fma_f32x2_rn_noftz(q_100, scale0_5, a_103);
                    acc_2[8] = r_107.x;
                    acc_2[9] = r_107.y;
                    float2 a_104 = make_float2(acc_2[10], acc_2[11]);
                    float2 q_101 = make_float2(tv_25[2], tv_25[3]);
                    float2 r_108 = fma_f32x2_rn_noftz(q_101, scale0_5, a_104);
                    acc_2[10] = r_108.x;
                    acc_2[11] = r_108.y;
                    float2 a_105 = make_float2(acc_2[12], acc_2[13]);
                    float2 q_102 = make_float2(tv_25[4], tv_25[5]);
                    float2 r_109 = fma_f32x2_rn_noftz(q_102, scale0_5, a_105);
                    acc_2[12] = r_109.x;
                    acc_2[13] = r_109.y;
                    float2 a_106 = make_float2(acc_2[14], acc_2[15]);
                    float2 q_103 = make_float2(tv_25[6], tv_25[7]);
                    float2 r_110 = fma_f32x2_rn_noftz(q_103, scale0_5, a_106);
                    acc_2[14] = r_110.x;
                    acc_2[15] = r_110.y;
                    float tv_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_101[2 + 0]));
                        (&tv_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_26[0]), "=f"(tv_26[1]), "=f"(tv_26[2]), "=f"(tv_26[3]), "=f"(tv_26[4]), "=f"(tv_26[5]), "=f"(tv_26[6]), "=f"(tv_26[7]) : "r"(r_101[2 + 0]));
                        #endif
                    }
                    float2 a_107 = make_float2(acc_2[16], acc_2[17]);
                    float2 q_104 = make_float2(tv_26[0], tv_26[1]);
                    float2 r_111 = fma_f32x2_rn_noftz(q_104, scale1_5, a_107);
                    acc_2[16] = r_111.x;
                    acc_2[17] = r_111.y;
                    float2 a_108 = make_float2(acc_2[18], acc_2[19]);
                    float2 q_105 = make_float2(tv_26[2], tv_26[3]);
                    float2 r_112 = fma_f32x2_rn_noftz(q_105, scale1_5, a_108);
                    acc_2[18] = r_112.x;
                    acc_2[19] = r_112.y;
                    float2 a_109 = make_float2(acc_2[20], acc_2[21]);
                    float2 q_106 = make_float2(tv_26[4], tv_26[5]);
                    float2 r_113 = fma_f32x2_rn_noftz(q_106, scale1_5, a_109);
                    acc_2[20] = r_113.x;
                    acc_2[21] = r_113.y;
                    float2 a_110 = make_float2(acc_2[22], acc_2[23]);
                    float2 q_107 = make_float2(tv_26[6], tv_26[7]);
                    float2 r_114 = fma_f32x2_rn_noftz(q_107, scale1_5, a_110);
                    acc_2[22] = r_114.x;
                    acc_2[23] = r_114.y;
                    float tv_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_101[3 + 0]));
                        (&tv_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_27[0]), "=f"(tv_27[1]), "=f"(tv_27[2]), "=f"(tv_27[3]), "=f"(tv_27[4]), "=f"(tv_27[5]), "=f"(tv_27[6]), "=f"(tv_27[7]) : "r"(r_101[3 + 0]));
                        #endif
                    }
                    float2 a_111 = make_float2(acc_2[24], acc_2[25]);
                    float2 q_108 = make_float2(tv_27[0], tv_27[1]);
                    float2 r_115 = fma_f32x2_rn_noftz(q_108, scale1_5, a_111);
                    acc_2[24] = r_115.x;
                    acc_2[25] = r_115.y;
                    float2 a_112 = make_float2(acc_2[26], acc_2[27]);
                    float2 q_109 = make_float2(tv_27[2], tv_27[3]);
                    float2 r_116 = fma_f32x2_rn_noftz(q_109, scale1_5, a_112);
                    acc_2[26] = r_116.x;
                    acc_2[27] = r_116.y;
                    float2 a_113 = make_float2(acc_2[28], acc_2[29]);
                    float2 q_110 = make_float2(tv_27[4], tv_27[5]);
                    float2 r_117 = fma_f32x2_rn_noftz(q_110, scale1_5, a_113);
                    acc_2[28] = r_117.x;
                    acc_2[29] = r_117.y;
                    float2 a_114 = make_float2(acc_2[30], acc_2[31]);
                    float2 q_111 = make_float2(tv_27[6], tv_27[7]);
                    float2 r_118 = fma_f32x2_rn_noftz(q_111, scale1_5, a_114);
                    acc_2[30] = r_118.x;
                    acc_2[31] = r_118.y;
                    uint32_t e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_6) : "h"((uint16_t)(b_sc_2)));
                    float cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_7) : "h"((uint16_t)(e4m3x2_to_f16x2_6)));
                    float cvt_f32_f16_1_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_6) : "h"((uint16_t)(e4m3x2_to_f16x2_6 >> 16)));
                    float s0_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_6) : "f"(cvt_f32_f16_7), "f"(rs_b_f_0));
                    float s1_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_6) : "f"(cvt_f32_f16_1_6), "f"(rs_b_f_0));
                    float lo_6 = s0_6;
                    float hi_6 = s1_6;
                    if (fhi_8) {
                        lo_6 = s1_6;
                        hi_6 = s0_6;
                    }
                    float2 scale0_6 = make_float2(lo_6, lo_6);
                    float2 scale1_6 = make_float2(hi_6, hi_6);
                    float tv_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_102[0]));
                        (&tv_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_28[0]), "=f"(tv_28[1]), "=f"(tv_28[2]), "=f"(tv_28[3]), "=f"(tv_28[4]), "=f"(tv_28[5]), "=f"(tv_28[6]), "=f"(tv_28[7]) : "r"(r_102[0]));
                        #endif
                    }
                    float2 a_115 = make_float2(acc_2[0], acc_2[1]);
                    float2 q_112 = make_float2(tv_28[0], tv_28[1]);
                    float2 r_119 = fma_f32x2_rn_noftz(q_112, scale0_6, a_115);
                    acc_2[0] = r_119.x;
                    acc_2[1] = r_119.y;
                    float2 a_116 = make_float2(acc_2[2], acc_2[3]);
                    float2 q_113 = make_float2(tv_28[2], tv_28[3]);
                    float2 r_120 = fma_f32x2_rn_noftz(q_113, scale0_6, a_116);
                    acc_2[2] = r_120.x;
                    acc_2[3] = r_120.y;
                    float2 a_117 = make_float2(acc_2[4], acc_2[5]);
                    float2 q_114 = make_float2(tv_28[4], tv_28[5]);
                    float2 r_121 = fma_f32x2_rn_noftz(q_114, scale0_6, a_117);
                    acc_2[4] = r_121.x;
                    acc_2[5] = r_121.y;
                    float2 a_118 = make_float2(acc_2[6], acc_2[7]);
                    float2 q_115 = make_float2(tv_28[6], tv_28[7]);
                    float2 r_122 = fma_f32x2_rn_noftz(q_115, scale0_6, a_118);
                    acc_2[6] = r_122.x;
                    acc_2[7] = r_122.y;
                    float tv_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_102[1 + 0]));
                        (&tv_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_29[0]), "=f"(tv_29[1]), "=f"(tv_29[2]), "=f"(tv_29[3]), "=f"(tv_29[4]), "=f"(tv_29[5]), "=f"(tv_29[6]), "=f"(tv_29[7]) : "r"(r_102[1 + 0]));
                        #endif
                    }
                    float2 a_119 = make_float2(acc_2[8], acc_2[9]);
                    float2 q_116 = make_float2(tv_29[0], tv_29[1]);
                    float2 r_123 = fma_f32x2_rn_noftz(q_116, scale0_6, a_119);
                    acc_2[8] = r_123.x;
                    acc_2[9] = r_123.y;
                    float2 a_120 = make_float2(acc_2[10], acc_2[11]);
                    float2 q_117 = make_float2(tv_29[2], tv_29[3]);
                    float2 r_124 = fma_f32x2_rn_noftz(q_117, scale0_6, a_120);
                    acc_2[10] = r_124.x;
                    acc_2[11] = r_124.y;
                    float2 a_121 = make_float2(acc_2[12], acc_2[13]);
                    float2 q_118 = make_float2(tv_29[4], tv_29[5]);
                    float2 r_125 = fma_f32x2_rn_noftz(q_118, scale0_6, a_121);
                    acc_2[12] = r_125.x;
                    acc_2[13] = r_125.y;
                    float2 a_122 = make_float2(acc_2[14], acc_2[15]);
                    float2 q_119 = make_float2(tv_29[6], tv_29[7]);
                    float2 r_126 = fma_f32x2_rn_noftz(q_119, scale0_6, a_122);
                    acc_2[14] = r_126.x;
                    acc_2[15] = r_126.y;
                    float tv_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_102[2 + 0]));
                        (&tv_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_30[0]), "=f"(tv_30[1]), "=f"(tv_30[2]), "=f"(tv_30[3]), "=f"(tv_30[4]), "=f"(tv_30[5]), "=f"(tv_30[6]), "=f"(tv_30[7]) : "r"(r_102[2 + 0]));
                        #endif
                    }
                    float2 a_123 = make_float2(acc_2[16], acc_2[17]);
                    float2 q_120 = make_float2(tv_30[0], tv_30[1]);
                    float2 r_127 = fma_f32x2_rn_noftz(q_120, scale1_6, a_123);
                    acc_2[16] = r_127.x;
                    acc_2[17] = r_127.y;
                    float2 a_124 = make_float2(acc_2[18], acc_2[19]);
                    float2 q_121 = make_float2(tv_30[2], tv_30[3]);
                    float2 r_128 = fma_f32x2_rn_noftz(q_121, scale1_6, a_124);
                    acc_2[18] = r_128.x;
                    acc_2[19] = r_128.y;
                    float2 a_125 = make_float2(acc_2[20], acc_2[21]);
                    float2 q_122 = make_float2(tv_30[4], tv_30[5]);
                    float2 r_129 = fma_f32x2_rn_noftz(q_122, scale1_6, a_125);
                    acc_2[20] = r_129.x;
                    acc_2[21] = r_129.y;
                    float2 a_126 = make_float2(acc_2[22], acc_2[23]);
                    float2 q_123 = make_float2(tv_30[6], tv_30[7]);
                    float2 r_130 = fma_f32x2_rn_noftz(q_123, scale1_6, a_126);
                    acc_2[22] = r_130.x;
                    acc_2[23] = r_130.y;
                    float tv_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_102[3 + 0]));
                        (&tv_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_31[0]), "=f"(tv_31[1]), "=f"(tv_31[2]), "=f"(tv_31[3]), "=f"(tv_31[4]), "=f"(tv_31[5]), "=f"(tv_31[6]), "=f"(tv_31[7]) : "r"(r_102[3 + 0]));
                        #endif
                    }
                    float2 a_127 = make_float2(acc_2[24], acc_2[25]);
                    float2 q_124 = make_float2(tv_31[0], tv_31[1]);
                    float2 r_131 = fma_f32x2_rn_noftz(q_124, scale1_6, a_127);
                    acc_2[24] = r_131.x;
                    acc_2[25] = r_131.y;
                    float2 a_128 = make_float2(acc_2[26], acc_2[27]);
                    float2 q_125 = make_float2(tv_31[2], tv_31[3]);
                    float2 r_132 = fma_f32x2_rn_noftz(q_125, scale1_6, a_128);
                    acc_2[26] = r_132.x;
                    acc_2[27] = r_132.y;
                    float2 a_129 = make_float2(acc_2[28], acc_2[29]);
                    float2 q_126 = make_float2(tv_31[4], tv_31[5]);
                    float2 r_133 = fma_f32x2_rn_noftz(q_126, scale1_6, a_129);
                    acc_2[28] = r_133.x;
                    acc_2[29] = r_133.y;
                    float2 a_130 = make_float2(acc_2[30], acc_2[31]);
                    float2 q_127 = make_float2(tv_31[6], tv_31[7]);
                    float2 r_134 = fma_f32x2_rn_noftz(q_127, scale1_6, a_130);
                    acc_2[30] = r_134.x;
                    acc_2[31] = r_134.y;
                    unsigned int t_2[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_2[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 3))));
                    float2 a2_1_48 = make_float2(acc_2[0], acc_2[1]);
                    float2 z_1_48 = add_f32x2_noftz(a2_1_48, make_float2(__uint_as_float(t_2[0] << 16), __uint_as_float(t_2[0] & 4294901760u)));
                    acc_2[0] = z_1_48.x;
                    acc_2[1] = z_1_48.y;
                    float2 ss2_1_48 = fma_f32x2_rn_noftz(z_1_48, z_1_48, make_float2(ssx, ssy));
                    ssx = ss2_1_48.x;
                    ssy = ss2_1_48.y;
                    float2 a2_1_49 = make_float2(acc_2[2], acc_2[3]);
                    float2 z_1_49 = add_f32x2_noftz(a2_1_49, make_float2(__uint_as_float(t_2[1] << 16), __uint_as_float(t_2[1] & 4294901760u)));
                    acc_2[2] = z_1_49.x;
                    acc_2[3] = z_1_49.y;
                    float2 ss2_1_49 = fma_f32x2_rn_noftz(z_1_49, z_1_49, make_float2(ssx, ssy));
                    ssx = ss2_1_49.x;
                    ssy = ss2_1_49.y;
                    float2 a2_1_50 = make_float2(acc_2[4], acc_2[5]);
                    float2 z_1_50 = add_f32x2_noftz(a2_1_50, make_float2(__uint_as_float(t_2[2] << 16), __uint_as_float(t_2[2] & 4294901760u)));
                    acc_2[4] = z_1_50.x;
                    acc_2[5] = z_1_50.y;
                    float2 ss2_1_50 = fma_f32x2_rn_noftz(z_1_50, z_1_50, make_float2(ssx, ssy));
                    ssx = ss2_1_50.x;
                    ssy = ss2_1_50.y;
                    float2 a2_1_51 = make_float2(acc_2[6], acc_2[7]);
                    float2 z_1_51 = add_f32x2_noftz(a2_1_51, make_float2(__uint_as_float(t_2[3] << 16), __uint_as_float(t_2[3] & 4294901760u)));
                    acc_2[6] = z_1_51.x;
                    acc_2[7] = z_1_51.y;
                    float2 ss2_1_51 = fma_f32x2_rn_noftz(z_1_51, z_1_51, make_float2(ssx, ssy));
                    ssx = ss2_1_51.x;
                    ssy = ss2_1_51.y;
                    float2 a2_1_52 = make_float2(acc_2[8], acc_2[9]);
                    float2 z_1_52 = add_f32x2_noftz(a2_1_52, make_float2(__uint_as_float(t_2[4] << 16), __uint_as_float(t_2[4] & 4294901760u)));
                    acc_2[8] = z_1_52.x;
                    acc_2[9] = z_1_52.y;
                    float2 ss2_1_52 = fma_f32x2_rn_noftz(z_1_52, z_1_52, make_float2(ssx, ssy));
                    ssx = ss2_1_52.x;
                    ssy = ss2_1_52.y;
                    float2 a2_1_53 = make_float2(acc_2[10], acc_2[11]);
                    float2 z_1_53 = add_f32x2_noftz(a2_1_53, make_float2(__uint_as_float(t_2[5] << 16), __uint_as_float(t_2[5] & 4294901760u)));
                    acc_2[10] = z_1_53.x;
                    acc_2[11] = z_1_53.y;
                    float2 ss2_1_53 = fma_f32x2_rn_noftz(z_1_53, z_1_53, make_float2(ssx, ssy));
                    ssx = ss2_1_53.x;
                    ssy = ss2_1_53.y;
                    float2 a2_1_54 = make_float2(acc_2[12], acc_2[13]);
                    float2 z_1_54 = add_f32x2_noftz(a2_1_54, make_float2(__uint_as_float(t_2[6] << 16), __uint_as_float(t_2[6] & 4294901760u)));
                    acc_2[12] = z_1_54.x;
                    acc_2[13] = z_1_54.y;
                    float2 ss2_1_54 = fma_f32x2_rn_noftz(z_1_54, z_1_54, make_float2(ssx, ssy));
                    ssx = ss2_1_54.x;
                    ssy = ss2_1_54.y;
                    float2 a2_1_55 = make_float2(acc_2[14], acc_2[15]);
                    float2 z_1_55 = add_f32x2_noftz(a2_1_55, make_float2(__uint_as_float(t_2[7] << 16), __uint_as_float(t_2[7] & 4294901760u)));
                    acc_2[14] = z_1_55.x;
                    acc_2[15] = z_1_55.y;
                    float2 ss2_1_55 = fma_f32x2_rn_noftz(z_1_55, z_1_55, make_float2(ssx, ssy));
                    ssx = ss2_1_55.x;
                    ssy = ss2_1_55.y;
                    float2 a2_1_56 = make_float2(acc_2[16], acc_2[17]);
                    float2 z_1_56 = add_f32x2_noftz(a2_1_56, make_float2(__uint_as_float(t_2[8] << 16), __uint_as_float(t_2[8] & 4294901760u)));
                    acc_2[16] = z_1_56.x;
                    acc_2[17] = z_1_56.y;
                    float2 ss2_1_56 = fma_f32x2_rn_noftz(z_1_56, z_1_56, make_float2(ssx, ssy));
                    ssx = ss2_1_56.x;
                    ssy = ss2_1_56.y;
                    float2 a2_1_57 = make_float2(acc_2[18], acc_2[19]);
                    float2 z_1_57 = add_f32x2_noftz(a2_1_57, make_float2(__uint_as_float(t_2[9] << 16), __uint_as_float(t_2[9] & 4294901760u)));
                    acc_2[18] = z_1_57.x;
                    acc_2[19] = z_1_57.y;
                    float2 ss2_1_57 = fma_f32x2_rn_noftz(z_1_57, z_1_57, make_float2(ssx, ssy));
                    ssx = ss2_1_57.x;
                    ssy = ss2_1_57.y;
                    float2 a2_1_58 = make_float2(acc_2[20], acc_2[21]);
                    float2 z_1_58 = add_f32x2_noftz(a2_1_58, make_float2(__uint_as_float(t_2[10] << 16), __uint_as_float(t_2[10] & 4294901760u)));
                    acc_2[20] = z_1_58.x;
                    acc_2[21] = z_1_58.y;
                    float2 ss2_1_58 = fma_f32x2_rn_noftz(z_1_58, z_1_58, make_float2(ssx, ssy));
                    ssx = ss2_1_58.x;
                    ssy = ss2_1_58.y;
                    float2 a2_1_59 = make_float2(acc_2[22], acc_2[23]);
                    float2 z_1_59 = add_f32x2_noftz(a2_1_59, make_float2(__uint_as_float(t_2[11] << 16), __uint_as_float(t_2[11] & 4294901760u)));
                    acc_2[22] = z_1_59.x;
                    acc_2[23] = z_1_59.y;
                    float2 ss2_1_59 = fma_f32x2_rn_noftz(z_1_59, z_1_59, make_float2(ssx, ssy));
                    ssx = ss2_1_59.x;
                    ssy = ss2_1_59.y;
                    float2 a2_1_60 = make_float2(acc_2[24], acc_2[25]);
                    float2 z_1_60 = add_f32x2_noftz(a2_1_60, make_float2(__uint_as_float(t_2[12] << 16), __uint_as_float(t_2[12] & 4294901760u)));
                    acc_2[24] = z_1_60.x;
                    acc_2[25] = z_1_60.y;
                    float2 ss2_1_60 = fma_f32x2_rn_noftz(z_1_60, z_1_60, make_float2(ssx, ssy));
                    ssx = ss2_1_60.x;
                    ssy = ss2_1_60.y;
                    float2 a2_1_61 = make_float2(acc_2[26], acc_2[27]);
                    float2 z_1_61 = add_f32x2_noftz(a2_1_61, make_float2(__uint_as_float(t_2[13] << 16), __uint_as_float(t_2[13] & 4294901760u)));
                    acc_2[26] = z_1_61.x;
                    acc_2[27] = z_1_61.y;
                    float2 ss2_1_61 = fma_f32x2_rn_noftz(z_1_61, z_1_61, make_float2(ssx, ssy));
                    ssx = ss2_1_61.x;
                    ssy = ss2_1_61.y;
                    float2 a2_1_62 = make_float2(acc_2[28], acc_2[29]);
                    float2 z_1_62 = add_f32x2_noftz(a2_1_62, make_float2(__uint_as_float(t_2[14] << 16), __uint_as_float(t_2[14] & 4294901760u)));
                    acc_2[28] = z_1_62.x;
                    acc_2[29] = z_1_62.y;
                    float2 ss2_1_62 = fma_f32x2_rn_noftz(z_1_62, z_1_62, make_float2(ssx, ssy));
                    ssx = ss2_1_62.x;
                    ssy = ss2_1_62.y;
                    float2 a2_1_63 = make_float2(acc_2[30], acc_2[31]);
                    float2 z_1_63 = add_f32x2_noftz(a2_1_63, make_float2(__uint_as_float(t_2[15] << 16), __uint_as_float(t_2[15] & 4294901760u)));
                    acc_2[30] = z_1_63.x;
                    acc_2[31] = z_1_63.y;
                    float2 ss2_1_63 = fma_f32x2_rn_noftz(z_1_63, z_1_63, make_float2(ssx, ssy));
                    ssx = ss2_1_63.x;
                    ssy = ss2_1_63.y;
                    unsigned int acc_bf16_2[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_2[_lp*2 + 0], acc_2[_lp*2+1 + 0]));
                        acc_bf16_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 0))), "r"(acc_bf16_2[0]), "r"(acc_bf16_2[1]), "r"(acc_bf16_2[2]), "r"(acc_bf16_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 1))), "r"(acc_bf16_2[4]), "r"(acc_bf16_2[5]), "r"(acc_bf16_2[6]), "r"(acc_bf16_2[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 2))), "r"(acc_bf16_2[8]), "r"(acc_bf16_2[9]), "r"(acc_bf16_2[10]), "r"(acc_bf16_2[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_3 + 16 * (f ^ 3))), "r"(acc_bf16_2[12]), "r"(acc_bf16_2[13]), "r"(acc_bf16_2[14]), "r"(acc_bf16_2[15]) : "memory");
                    unsigned int d_1 = 0;
                    d_1 = d_1 | pv[16] & ~((pv[16] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[17] & ~((pv[17] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[18] & ~((pv[18] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[19] & ~((pv[19] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | ((ps[4] & 255) + 1 | (ps[4] >> 8) + 1) & 256;
                    d_1 = d_1 | pv[20] & ~((pv[20] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[21] & ~((pv[21] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[22] & ~((pv[22] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | pv[23] & ~((pv[23] & 2004318071) + 2004318071) & 2290649224u;
                    d_1 = d_1 | ((ps[5] & 255) + 1 | (ps[5] >> 8) + 1) & 256;
                    if (d_1 != 0) {
                        int pp0_2 = lane_2 + 128;
                        int pp1_2 = pp0_2 + 32;
                        uint32_t fp4_wire_values_1[8];
                        uint32_t fp4_wire_scales_1[2];
                        uint16_t _wire_h16_7_0;
                        uint16_t _wire_h16_7_1;
                        uint32_t _wire_vpend_7 = 3u;
                        uint32_t _wire_spend_7 = 3u;
                        do {
                            if (_wire_vpend_7 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[0]), "=r"(fp4_wire_values_1[1]), "=r"(fp4_wire_values_1[2]), "=r"(fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0_2)))) : "memory");
                            }
                            if (_wire_vpend_7 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(fp4_wire_values_1[4]), "=r"(fp4_wire_values_1[5]), "=r"(fp4_wire_values_1[6]), "=r"(fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1_2)))) : "memory");
                            }
                            if (_wire_spend_7 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_7_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0_2)))) : "memory");
                                fp4_wire_scales_1[0] = (uint32_t)_wire_h16_7_0;
                            }
                            if (_wire_spend_7 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_7_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1_2)))) : "memory");
                                fp4_wire_scales_1[1] = (uint32_t)_wire_h16_7_1;
                            }
                            _wire_vpend_7 = ((uint32_t)((((fp4_wire_values_1[0] & ~((fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[1] & ~((fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[2] & ~((fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[3] & ~((fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((fp4_wire_values_1[4] & ~((fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[5] & ~((fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[6] & ~((fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((fp4_wire_values_1[7] & ~((fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_7 = ((uint32_t)(((uint32_t)((fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_7 | _wire_spend_7) != 0u);
                        pv[16] = fp4_wire_values_1[0];
                        pv[17] = fp4_wire_values_1[1];
                        pv[18] = fp4_wire_values_1[2];
                        pv[19] = fp4_wire_values_1[3];
                        ps[4] = fp4_wire_scales_1[0];
                        pv[20] = fp4_wire_values_1[4];
                        pv[21] = fp4_wire_values_1[5];
                        pv[22] = fp4_wire_values_1[6];
                        pv[23] = fp4_wire_values_1[7];
                        ps[5] = fp4_wire_scales_1[1];
                    }
                    int pp_4 = lane_2 + 128;
                    unsigned int own_3[4];
                    unsigned int own_sc_3[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_4)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_3[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_4)));
                    unsigned int a_131[4];
                    unsigned int b_3[4];
                    a_131[0] = ((first) ? own_3[0] : pv[16]);
                    b_3[0] = ((first) ? pv[16] : own_3[0]);
                    a_131[1] = ((first) ? own_3[1] : pv[17]);
                    b_3[1] = ((first) ? pv[17] : own_3[1]);
                    a_131[2] = ((first) ? own_3[2] : pv[18]);
                    b_3[2] = ((first) ? pv[18] : own_3[2]);
                    a_131[3] = ((first) ? own_3[3] : pv[19]);
                    b_3[3] = ((first) ? pv[19] : own_3[3]);
                    unsigned int a_sc_3 = ((first) ? own_sc_3[0] : ps[4]);
                    unsigned int b_sc_3 = ((first) ? ps[4] : own_sc_3[0]);
                    float acc_3[32] = {0};
                    bool fhi_11 = (f & 2) != 0;
                    bool fodd_7 = (f & 1) != 0;
                    bool fhi_12 = (f & 2) != 0;
                    unsigned int s_7[4];
                    s_7[0] = a_131[0];
                    s_7[1] = a_131[1];
                    s_7[2] = a_131[2];
                    s_7[3] = a_131[3];
                    if (fodd_7) {
                        s_7[0] = a_131[1];
                        s_7[1] = a_131[0];
                        s_7[2] = a_131[3];
                        s_7[3] = a_131[2];
                    }
                    unsigned int r_135[4];
                    r_135[0] = s_7[0];
                    r_135[1] = s_7[1];
                    r_135[2] = s_7[2];
                    r_135[3] = s_7[3];
                    if (fhi_12) {
                        r_135[0] = s_7[2];
                        r_135[1] = s_7[3];
                        r_135[2] = s_7[0];
                        r_135[3] = s_7[1];
                    }
                    bool fodd_8 = (f & 1) != 0;
                    bool fhi_13 = (f & 2) != 0;
                    unsigned int s_8[4];
                    s_8[0] = b_3[0];
                    s_8[1] = b_3[1];
                    s_8[2] = b_3[2];
                    s_8[3] = b_3[3];
                    if (fodd_8) {
                        s_8[0] = b_3[1];
                        s_8[1] = b_3[0];
                        s_8[2] = b_3[3];
                        s_8[3] = b_3[2];
                    }
                    unsigned int r_136[4];
                    r_136[0] = s_8[0];
                    r_136[1] = s_8[1];
                    r_136[2] = s_8[2];
                    r_136[3] = s_8[3];
                    if (fhi_13) {
                        r_136[0] = s_8[2];
                        r_136[1] = s_8[3];
                        r_136[2] = s_8[0];
                        r_136[3] = s_8[1];
                    }
                    uint32_t e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_7) : "h"((uint16_t)(a_sc_3)));
                    float cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_8) : "h"((uint16_t)(e4m3x2_to_f16x2_7)));
                    float cvt_f32_f16_1_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_7) : "h"((uint16_t)(e4m3x2_to_f16x2_7 >> 16)));
                    float s0_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_7) : "f"(cvt_f32_f16_8), "f"(rs_a_f_0));
                    float s1_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_7) : "f"(cvt_f32_f16_1_7), "f"(rs_a_f_0));
                    float lo_7 = s0_7;
                    float hi_7 = s1_7;
                    if (fhi_11) {
                        lo_7 = s1_7;
                        hi_7 = s0_7;
                    }
                    float2 scale0_7 = make_float2(lo_7, lo_7);
                    float2 scale1_7 = make_float2(hi_7, hi_7);
                    float tv_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_135[0]));
                        (&tv_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_32[0]), "=f"(tv_32[1]), "=f"(tv_32[2]), "=f"(tv_32[3]), "=f"(tv_32[4]), "=f"(tv_32[5]), "=f"(tv_32[6]), "=f"(tv_32[7]) : "r"(r_135[0]));
                        #endif
                    }
                    float2 a_132 = make_float2(acc_3[0], acc_3[1]);
                    float2 q_128 = make_float2(tv_32[0], tv_32[1]);
                    float2 r_137 = fma_f32x2_rn_noftz(q_128, scale0_7, a_132);
                    acc_3[0] = r_137.x;
                    acc_3[1] = r_137.y;
                    float2 a_133 = make_float2(acc_3[2], acc_3[3]);
                    float2 q_129 = make_float2(tv_32[2], tv_32[3]);
                    float2 r_138 = fma_f32x2_rn_noftz(q_129, scale0_7, a_133);
                    acc_3[2] = r_138.x;
                    acc_3[3] = r_138.y;
                    float2 a_134 = make_float2(acc_3[4], acc_3[5]);
                    float2 q_130 = make_float2(tv_32[4], tv_32[5]);
                    float2 r_139 = fma_f32x2_rn_noftz(q_130, scale0_7, a_134);
                    acc_3[4] = r_139.x;
                    acc_3[5] = r_139.y;
                    float2 a_135 = make_float2(acc_3[6], acc_3[7]);
                    float2 q_131 = make_float2(tv_32[6], tv_32[7]);
                    float2 r_140 = fma_f32x2_rn_noftz(q_131, scale0_7, a_135);
                    acc_3[6] = r_140.x;
                    acc_3[7] = r_140.y;
                    float tv_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_135[1 + 0]));
                        (&tv_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_33[0]), "=f"(tv_33[1]), "=f"(tv_33[2]), "=f"(tv_33[3]), "=f"(tv_33[4]), "=f"(tv_33[5]), "=f"(tv_33[6]), "=f"(tv_33[7]) : "r"(r_135[1 + 0]));
                        #endif
                    }
                    float2 a_136 = make_float2(acc_3[8], acc_3[9]);
                    float2 q_132 = make_float2(tv_33[0], tv_33[1]);
                    float2 r_141 = fma_f32x2_rn_noftz(q_132, scale0_7, a_136);
                    acc_3[8] = r_141.x;
                    acc_3[9] = r_141.y;
                    float2 a_137 = make_float2(acc_3[10], acc_3[11]);
                    float2 q_133 = make_float2(tv_33[2], tv_33[3]);
                    float2 r_142 = fma_f32x2_rn_noftz(q_133, scale0_7, a_137);
                    acc_3[10] = r_142.x;
                    acc_3[11] = r_142.y;
                    float2 a_138 = make_float2(acc_3[12], acc_3[13]);
                    float2 q_134 = make_float2(tv_33[4], tv_33[5]);
                    float2 r_143 = fma_f32x2_rn_noftz(q_134, scale0_7, a_138);
                    acc_3[12] = r_143.x;
                    acc_3[13] = r_143.y;
                    float2 a_139 = make_float2(acc_3[14], acc_3[15]);
                    float2 q_135 = make_float2(tv_33[6], tv_33[7]);
                    float2 r_144 = fma_f32x2_rn_noftz(q_135, scale0_7, a_139);
                    acc_3[14] = r_144.x;
                    acc_3[15] = r_144.y;
                    float tv_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_135[2 + 0]));
                        (&tv_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_34[0]), "=f"(tv_34[1]), "=f"(tv_34[2]), "=f"(tv_34[3]), "=f"(tv_34[4]), "=f"(tv_34[5]), "=f"(tv_34[6]), "=f"(tv_34[7]) : "r"(r_135[2 + 0]));
                        #endif
                    }
                    float2 a_140 = make_float2(acc_3[16], acc_3[17]);
                    float2 q_136 = make_float2(tv_34[0], tv_34[1]);
                    float2 r_145 = fma_f32x2_rn_noftz(q_136, scale1_7, a_140);
                    acc_3[16] = r_145.x;
                    acc_3[17] = r_145.y;
                    float2 a_141 = make_float2(acc_3[18], acc_3[19]);
                    float2 q_137 = make_float2(tv_34[2], tv_34[3]);
                    float2 r_146 = fma_f32x2_rn_noftz(q_137, scale1_7, a_141);
                    acc_3[18] = r_146.x;
                    acc_3[19] = r_146.y;
                    float2 a_142 = make_float2(acc_3[20], acc_3[21]);
                    float2 q_138 = make_float2(tv_34[4], tv_34[5]);
                    float2 r_147 = fma_f32x2_rn_noftz(q_138, scale1_7, a_142);
                    acc_3[20] = r_147.x;
                    acc_3[21] = r_147.y;
                    float2 a_143 = make_float2(acc_3[22], acc_3[23]);
                    float2 q_139 = make_float2(tv_34[6], tv_34[7]);
                    float2 r_148 = fma_f32x2_rn_noftz(q_139, scale1_7, a_143);
                    acc_3[22] = r_148.x;
                    acc_3[23] = r_148.y;
                    float tv_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_135[3 + 0]));
                        (&tv_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_35[0]), "=f"(tv_35[1]), "=f"(tv_35[2]), "=f"(tv_35[3]), "=f"(tv_35[4]), "=f"(tv_35[5]), "=f"(tv_35[6]), "=f"(tv_35[7]) : "r"(r_135[3 + 0]));
                        #endif
                    }
                    float2 a_144 = make_float2(acc_3[24], acc_3[25]);
                    float2 q_140 = make_float2(tv_35[0], tv_35[1]);
                    float2 r_149 = fma_f32x2_rn_noftz(q_140, scale1_7, a_144);
                    acc_3[24] = r_149.x;
                    acc_3[25] = r_149.y;
                    float2 a_145 = make_float2(acc_3[26], acc_3[27]);
                    float2 q_141 = make_float2(tv_35[2], tv_35[3]);
                    float2 r_150 = fma_f32x2_rn_noftz(q_141, scale1_7, a_145);
                    acc_3[26] = r_150.x;
                    acc_3[27] = r_150.y;
                    float2 a_146 = make_float2(acc_3[28], acc_3[29]);
                    float2 q_142 = make_float2(tv_35[4], tv_35[5]);
                    float2 r_151 = fma_f32x2_rn_noftz(q_142, scale1_7, a_146);
                    acc_3[28] = r_151.x;
                    acc_3[29] = r_151.y;
                    float2 a_147 = make_float2(acc_3[30], acc_3[31]);
                    float2 q_143 = make_float2(tv_35[6], tv_35[7]);
                    float2 r_152 = fma_f32x2_rn_noftz(q_143, scale1_7, a_147);
                    acc_3[30] = r_152.x;
                    acc_3[31] = r_152.y;
                    uint32_t e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_8) : "h"((uint16_t)(b_sc_3)));
                    float cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_9) : "h"((uint16_t)(e4m3x2_to_f16x2_8)));
                    float cvt_f32_f16_1_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_8) : "h"((uint16_t)(e4m3x2_to_f16x2_8 >> 16)));
                    float s0_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_8) : "f"(cvt_f32_f16_9), "f"(rs_b_f_0));
                    float s1_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_8) : "f"(cvt_f32_f16_1_8), "f"(rs_b_f_0));
                    float lo_8 = s0_8;
                    float hi_8 = s1_8;
                    if (fhi_11) {
                        lo_8 = s1_8;
                        hi_8 = s0_8;
                    }
                    float2 scale0_8 = make_float2(lo_8, lo_8);
                    float2 scale1_8 = make_float2(hi_8, hi_8);
                    float tv_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_136[0]));
                        (&tv_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_36[0]), "=f"(tv_36[1]), "=f"(tv_36[2]), "=f"(tv_36[3]), "=f"(tv_36[4]), "=f"(tv_36[5]), "=f"(tv_36[6]), "=f"(tv_36[7]) : "r"(r_136[0]));
                        #endif
                    }
                    float2 a_148 = make_float2(acc_3[0], acc_3[1]);
                    float2 q_144 = make_float2(tv_36[0], tv_36[1]);
                    float2 r_153 = fma_f32x2_rn_noftz(q_144, scale0_8, a_148);
                    acc_3[0] = r_153.x;
                    acc_3[1] = r_153.y;
                    float2 a_149 = make_float2(acc_3[2], acc_3[3]);
                    float2 q_145 = make_float2(tv_36[2], tv_36[3]);
                    float2 r_154 = fma_f32x2_rn_noftz(q_145, scale0_8, a_149);
                    acc_3[2] = r_154.x;
                    acc_3[3] = r_154.y;
                    float2 a_150 = make_float2(acc_3[4], acc_3[5]);
                    float2 q_146 = make_float2(tv_36[4], tv_36[5]);
                    float2 r_155 = fma_f32x2_rn_noftz(q_146, scale0_8, a_150);
                    acc_3[4] = r_155.x;
                    acc_3[5] = r_155.y;
                    float2 a_151 = make_float2(acc_3[6], acc_3[7]);
                    float2 q_147 = make_float2(tv_36[6], tv_36[7]);
                    float2 r_156 = fma_f32x2_rn_noftz(q_147, scale0_8, a_151);
                    acc_3[6] = r_156.x;
                    acc_3[7] = r_156.y;
                    float tv_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_136[1 + 0]));
                        (&tv_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_37[0]), "=f"(tv_37[1]), "=f"(tv_37[2]), "=f"(tv_37[3]), "=f"(tv_37[4]), "=f"(tv_37[5]), "=f"(tv_37[6]), "=f"(tv_37[7]) : "r"(r_136[1 + 0]));
                        #endif
                    }
                    float2 a_152 = make_float2(acc_3[8], acc_3[9]);
                    float2 q_148 = make_float2(tv_37[0], tv_37[1]);
                    float2 r_157 = fma_f32x2_rn_noftz(q_148, scale0_8, a_152);
                    acc_3[8] = r_157.x;
                    acc_3[9] = r_157.y;
                    float2 a_153 = make_float2(acc_3[10], acc_3[11]);
                    float2 q_149 = make_float2(tv_37[2], tv_37[3]);
                    float2 r_158 = fma_f32x2_rn_noftz(q_149, scale0_8, a_153);
                    acc_3[10] = r_158.x;
                    acc_3[11] = r_158.y;
                    float2 a_154 = make_float2(acc_3[12], acc_3[13]);
                    float2 q_150 = make_float2(tv_37[4], tv_37[5]);
                    float2 r_159 = fma_f32x2_rn_noftz(q_150, scale0_8, a_154);
                    acc_3[12] = r_159.x;
                    acc_3[13] = r_159.y;
                    float2 a_155 = make_float2(acc_3[14], acc_3[15]);
                    float2 q_151 = make_float2(tv_37[6], tv_37[7]);
                    float2 r_160 = fma_f32x2_rn_noftz(q_151, scale0_8, a_155);
                    acc_3[14] = r_160.x;
                    acc_3[15] = r_160.y;
                    float tv_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_136[2 + 0]));
                        (&tv_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_38[0]), "=f"(tv_38[1]), "=f"(tv_38[2]), "=f"(tv_38[3]), "=f"(tv_38[4]), "=f"(tv_38[5]), "=f"(tv_38[6]), "=f"(tv_38[7]) : "r"(r_136[2 + 0]));
                        #endif
                    }
                    float2 a_156 = make_float2(acc_3[16], acc_3[17]);
                    float2 q_152 = make_float2(tv_38[0], tv_38[1]);
                    float2 r_161 = fma_f32x2_rn_noftz(q_152, scale1_8, a_156);
                    acc_3[16] = r_161.x;
                    acc_3[17] = r_161.y;
                    float2 a_157 = make_float2(acc_3[18], acc_3[19]);
                    float2 q_153 = make_float2(tv_38[2], tv_38[3]);
                    float2 r_162 = fma_f32x2_rn_noftz(q_153, scale1_8, a_157);
                    acc_3[18] = r_162.x;
                    acc_3[19] = r_162.y;
                    float2 a_158 = make_float2(acc_3[20], acc_3[21]);
                    float2 q_154 = make_float2(tv_38[4], tv_38[5]);
                    float2 r_163 = fma_f32x2_rn_noftz(q_154, scale1_8, a_158);
                    acc_3[20] = r_163.x;
                    acc_3[21] = r_163.y;
                    float2 a_159 = make_float2(acc_3[22], acc_3[23]);
                    float2 q_155 = make_float2(tv_38[6], tv_38[7]);
                    float2 r_164 = fma_f32x2_rn_noftz(q_155, scale1_8, a_159);
                    acc_3[22] = r_164.x;
                    acc_3[23] = r_164.y;
                    float tv_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_136[3 + 0]));
                        (&tv_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_39[0]), "=f"(tv_39[1]), "=f"(tv_39[2]), "=f"(tv_39[3]), "=f"(tv_39[4]), "=f"(tv_39[5]), "=f"(tv_39[6]), "=f"(tv_39[7]) : "r"(r_136[3 + 0]));
                        #endif
                    }
                    float2 a_160 = make_float2(acc_3[24], acc_3[25]);
                    float2 q_156 = make_float2(tv_39[0], tv_39[1]);
                    float2 r_165 = fma_f32x2_rn_noftz(q_156, scale1_8, a_160);
                    acc_3[24] = r_165.x;
                    acc_3[25] = r_165.y;
                    float2 a_161 = make_float2(acc_3[26], acc_3[27]);
                    float2 q_157 = make_float2(tv_39[2], tv_39[3]);
                    float2 r_166 = fma_f32x2_rn_noftz(q_157, scale1_8, a_161);
                    acc_3[26] = r_166.x;
                    acc_3[27] = r_166.y;
                    float2 a_162 = make_float2(acc_3[28], acc_3[29]);
                    float2 q_158 = make_float2(tv_39[4], tv_39[5]);
                    float2 r_167 = fma_f32x2_rn_noftz(q_158, scale1_8, a_162);
                    acc_3[28] = r_167.x;
                    acc_3[29] = r_167.y;
                    float2 a_163 = make_float2(acc_3[30], acc_3[31]);
                    float2 q_159 = make_float2(tv_39[6], tv_39[7]);
                    float2 r_168 = fma_f32x2_rn_noftz(q_159, scale1_8, a_163);
                    acc_3[30] = r_168.x;
                    acc_3[31] = r_168.y;
                    unsigned int t_3[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_3[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 3))));
                    float2 a2_1_64 = make_float2(acc_3[0], acc_3[1]);
                    float2 z_1_64 = add_f32x2_noftz(a2_1_64, make_float2(__uint_as_float(t_3[0] << 16), __uint_as_float(t_3[0] & 4294901760u)));
                    acc_3[0] = z_1_64.x;
                    acc_3[1] = z_1_64.y;
                    float2 ss2_1_64 = fma_f32x2_rn_noftz(z_1_64, z_1_64, make_float2(ssx, ssy));
                    ssx = ss2_1_64.x;
                    ssy = ss2_1_64.y;
                    float2 a2_1_65 = make_float2(acc_3[2], acc_3[3]);
                    float2 z_1_65 = add_f32x2_noftz(a2_1_65, make_float2(__uint_as_float(t_3[1] << 16), __uint_as_float(t_3[1] & 4294901760u)));
                    acc_3[2] = z_1_65.x;
                    acc_3[3] = z_1_65.y;
                    float2 ss2_1_65 = fma_f32x2_rn_noftz(z_1_65, z_1_65, make_float2(ssx, ssy));
                    ssx = ss2_1_65.x;
                    ssy = ss2_1_65.y;
                    float2 a2_1_66 = make_float2(acc_3[4], acc_3[5]);
                    float2 z_1_66 = add_f32x2_noftz(a2_1_66, make_float2(__uint_as_float(t_3[2] << 16), __uint_as_float(t_3[2] & 4294901760u)));
                    acc_3[4] = z_1_66.x;
                    acc_3[5] = z_1_66.y;
                    float2 ss2_1_66 = fma_f32x2_rn_noftz(z_1_66, z_1_66, make_float2(ssx, ssy));
                    ssx = ss2_1_66.x;
                    ssy = ss2_1_66.y;
                    float2 a2_1_67 = make_float2(acc_3[6], acc_3[7]);
                    float2 z_1_67 = add_f32x2_noftz(a2_1_67, make_float2(__uint_as_float(t_3[3] << 16), __uint_as_float(t_3[3] & 4294901760u)));
                    acc_3[6] = z_1_67.x;
                    acc_3[7] = z_1_67.y;
                    float2 ss2_1_67 = fma_f32x2_rn_noftz(z_1_67, z_1_67, make_float2(ssx, ssy));
                    ssx = ss2_1_67.x;
                    ssy = ss2_1_67.y;
                    float2 a2_1_68 = make_float2(acc_3[8], acc_3[9]);
                    float2 z_1_68 = add_f32x2_noftz(a2_1_68, make_float2(__uint_as_float(t_3[4] << 16), __uint_as_float(t_3[4] & 4294901760u)));
                    acc_3[8] = z_1_68.x;
                    acc_3[9] = z_1_68.y;
                    float2 ss2_1_68 = fma_f32x2_rn_noftz(z_1_68, z_1_68, make_float2(ssx, ssy));
                    ssx = ss2_1_68.x;
                    ssy = ss2_1_68.y;
                    float2 a2_1_69 = make_float2(acc_3[10], acc_3[11]);
                    float2 z_1_69 = add_f32x2_noftz(a2_1_69, make_float2(__uint_as_float(t_3[5] << 16), __uint_as_float(t_3[5] & 4294901760u)));
                    acc_3[10] = z_1_69.x;
                    acc_3[11] = z_1_69.y;
                    float2 ss2_1_69 = fma_f32x2_rn_noftz(z_1_69, z_1_69, make_float2(ssx, ssy));
                    ssx = ss2_1_69.x;
                    ssy = ss2_1_69.y;
                    float2 a2_1_70 = make_float2(acc_3[12], acc_3[13]);
                    float2 z_1_70 = add_f32x2_noftz(a2_1_70, make_float2(__uint_as_float(t_3[6] << 16), __uint_as_float(t_3[6] & 4294901760u)));
                    acc_3[12] = z_1_70.x;
                    acc_3[13] = z_1_70.y;
                    float2 ss2_1_70 = fma_f32x2_rn_noftz(z_1_70, z_1_70, make_float2(ssx, ssy));
                    ssx = ss2_1_70.x;
                    ssy = ss2_1_70.y;
                    float2 a2_1_71 = make_float2(acc_3[14], acc_3[15]);
                    float2 z_1_71 = add_f32x2_noftz(a2_1_71, make_float2(__uint_as_float(t_3[7] << 16), __uint_as_float(t_3[7] & 4294901760u)));
                    acc_3[14] = z_1_71.x;
                    acc_3[15] = z_1_71.y;
                    float2 ss2_1_71 = fma_f32x2_rn_noftz(z_1_71, z_1_71, make_float2(ssx, ssy));
                    ssx = ss2_1_71.x;
                    ssy = ss2_1_71.y;
                    float2 a2_1_72 = make_float2(acc_3[16], acc_3[17]);
                    float2 z_1_72 = add_f32x2_noftz(a2_1_72, make_float2(__uint_as_float(t_3[8] << 16), __uint_as_float(t_3[8] & 4294901760u)));
                    acc_3[16] = z_1_72.x;
                    acc_3[17] = z_1_72.y;
                    float2 ss2_1_72 = fma_f32x2_rn_noftz(z_1_72, z_1_72, make_float2(ssx, ssy));
                    ssx = ss2_1_72.x;
                    ssy = ss2_1_72.y;
                    float2 a2_1_73 = make_float2(acc_3[18], acc_3[19]);
                    float2 z_1_73 = add_f32x2_noftz(a2_1_73, make_float2(__uint_as_float(t_3[9] << 16), __uint_as_float(t_3[9] & 4294901760u)));
                    acc_3[18] = z_1_73.x;
                    acc_3[19] = z_1_73.y;
                    float2 ss2_1_73 = fma_f32x2_rn_noftz(z_1_73, z_1_73, make_float2(ssx, ssy));
                    ssx = ss2_1_73.x;
                    ssy = ss2_1_73.y;
                    float2 a2_1_74 = make_float2(acc_3[20], acc_3[21]);
                    float2 z_1_74 = add_f32x2_noftz(a2_1_74, make_float2(__uint_as_float(t_3[10] << 16), __uint_as_float(t_3[10] & 4294901760u)));
                    acc_3[20] = z_1_74.x;
                    acc_3[21] = z_1_74.y;
                    float2 ss2_1_74 = fma_f32x2_rn_noftz(z_1_74, z_1_74, make_float2(ssx, ssy));
                    ssx = ss2_1_74.x;
                    ssy = ss2_1_74.y;
                    float2 a2_1_75 = make_float2(acc_3[22], acc_3[23]);
                    float2 z_1_75 = add_f32x2_noftz(a2_1_75, make_float2(__uint_as_float(t_3[11] << 16), __uint_as_float(t_3[11] & 4294901760u)));
                    acc_3[22] = z_1_75.x;
                    acc_3[23] = z_1_75.y;
                    float2 ss2_1_75 = fma_f32x2_rn_noftz(z_1_75, z_1_75, make_float2(ssx, ssy));
                    ssx = ss2_1_75.x;
                    ssy = ss2_1_75.y;
                    float2 a2_1_76 = make_float2(acc_3[24], acc_3[25]);
                    float2 z_1_76 = add_f32x2_noftz(a2_1_76, make_float2(__uint_as_float(t_3[12] << 16), __uint_as_float(t_3[12] & 4294901760u)));
                    acc_3[24] = z_1_76.x;
                    acc_3[25] = z_1_76.y;
                    float2 ss2_1_76 = fma_f32x2_rn_noftz(z_1_76, z_1_76, make_float2(ssx, ssy));
                    ssx = ss2_1_76.x;
                    ssy = ss2_1_76.y;
                    float2 a2_1_77 = make_float2(acc_3[26], acc_3[27]);
                    float2 z_1_77 = add_f32x2_noftz(a2_1_77, make_float2(__uint_as_float(t_3[13] << 16), __uint_as_float(t_3[13] & 4294901760u)));
                    acc_3[26] = z_1_77.x;
                    acc_3[27] = z_1_77.y;
                    float2 ss2_1_77 = fma_f32x2_rn_noftz(z_1_77, z_1_77, make_float2(ssx, ssy));
                    ssx = ss2_1_77.x;
                    ssy = ss2_1_77.y;
                    float2 a2_1_78 = make_float2(acc_3[28], acc_3[29]);
                    float2 z_1_78 = add_f32x2_noftz(a2_1_78, make_float2(__uint_as_float(t_3[14] << 16), __uint_as_float(t_3[14] & 4294901760u)));
                    acc_3[28] = z_1_78.x;
                    acc_3[29] = z_1_78.y;
                    float2 ss2_1_78 = fma_f32x2_rn_noftz(z_1_78, z_1_78, make_float2(ssx, ssy));
                    ssx = ss2_1_78.x;
                    ssy = ss2_1_78.y;
                    float2 a2_1_79 = make_float2(acc_3[30], acc_3[31]);
                    float2 z_1_79 = add_f32x2_noftz(a2_1_79, make_float2(__uint_as_float(t_3[15] << 16), __uint_as_float(t_3[15] & 4294901760u)));
                    acc_3[30] = z_1_79.x;
                    acc_3[31] = z_1_79.y;
                    float2 ss2_1_79 = fma_f32x2_rn_noftz(z_1_79, z_1_79, make_float2(ssx, ssy));
                    ssx = ss2_1_79.x;
                    ssy = ss2_1_79.y;
                    unsigned int acc_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_3[_lp*2 + 0], acc_3[_lp*2+1 + 0]));
                        acc_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 0))), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 1))), "r"(acc_bf16_3[4]), "r"(acc_bf16_3[5]), "r"(acc_bf16_3[6]), "r"(acc_bf16_3[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 2))), "r"(acc_bf16_3[8]), "r"(acc_bf16_3[9]), "r"(acc_bf16_3[10]), "r"(acc_bf16_3[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_4 + 16 * (f ^ 3))), "r"(acc_bf16_3[12]), "r"(acc_bf16_3[13]), "r"(acc_bf16_3[14]), "r"(acc_bf16_3[15]) : "memory");
                    int pp_5 = lane_2 + 160;
                    unsigned int own_4[4];
                    unsigned int own_sc_4[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_5)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_4[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_5)));
                    unsigned int a_164[4];
                    unsigned int b_4[4];
                    a_164[0] = ((first) ? own_4[0] : pv[20]);
                    b_4[0] = ((first) ? pv[20] : own_4[0]);
                    a_164[1] = ((first) ? own_4[1] : pv[21]);
                    b_4[1] = ((first) ? pv[21] : own_4[1]);
                    a_164[2] = ((first) ? own_4[2] : pv[22]);
                    b_4[2] = ((first) ? pv[22] : own_4[2]);
                    a_164[3] = ((first) ? own_4[3] : pv[23]);
                    b_4[3] = ((first) ? pv[23] : own_4[3]);
                    unsigned int a_sc_4 = ((first) ? own_sc_4[0] : ps[5]);
                    unsigned int b_sc_4 = ((first) ? ps[5] : own_sc_4[0]);
                    float acc_4[32] = {0};
                    bool fhi_14 = (f & 2) != 0;
                    bool fodd_9 = (f & 1) != 0;
                    bool fhi_15 = (f & 2) != 0;
                    unsigned int s_9[4];
                    s_9[0] = a_164[0];
                    s_9[1] = a_164[1];
                    s_9[2] = a_164[2];
                    s_9[3] = a_164[3];
                    if (fodd_9) {
                        s_9[0] = a_164[1];
                        s_9[1] = a_164[0];
                        s_9[2] = a_164[3];
                        s_9[3] = a_164[2];
                    }
                    unsigned int r_169[4];
                    r_169[0] = s_9[0];
                    r_169[1] = s_9[1];
                    r_169[2] = s_9[2];
                    r_169[3] = s_9[3];
                    if (fhi_15) {
                        r_169[0] = s_9[2];
                        r_169[1] = s_9[3];
                        r_169[2] = s_9[0];
                        r_169[3] = s_9[1];
                    }
                    bool fodd_10 = (f & 1) != 0;
                    bool fhi_16 = (f & 2) != 0;
                    unsigned int s_10[4];
                    s_10[0] = b_4[0];
                    s_10[1] = b_4[1];
                    s_10[2] = b_4[2];
                    s_10[3] = b_4[3];
                    if (fodd_10) {
                        s_10[0] = b_4[1];
                        s_10[1] = b_4[0];
                        s_10[2] = b_4[3];
                        s_10[3] = b_4[2];
                    }
                    unsigned int r_170[4];
                    r_170[0] = s_10[0];
                    r_170[1] = s_10[1];
                    r_170[2] = s_10[2];
                    r_170[3] = s_10[3];
                    if (fhi_16) {
                        r_170[0] = s_10[2];
                        r_170[1] = s_10[3];
                        r_170[2] = s_10[0];
                        r_170[3] = s_10[1];
                    }
                    uint32_t e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_9) : "h"((uint16_t)(a_sc_4)));
                    float cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_10) : "h"((uint16_t)(e4m3x2_to_f16x2_9)));
                    float cvt_f32_f16_1_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_9) : "h"((uint16_t)(e4m3x2_to_f16x2_9 >> 16)));
                    float s0_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_9) : "f"(cvt_f32_f16_10), "f"(rs_a_f_0));
                    float s1_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_9) : "f"(cvt_f32_f16_1_9), "f"(rs_a_f_0));
                    float lo_9 = s0_9;
                    float hi_9 = s1_9;
                    if (fhi_14) {
                        lo_9 = s1_9;
                        hi_9 = s0_9;
                    }
                    float2 scale0_9 = make_float2(lo_9, lo_9);
                    float2 scale1_9 = make_float2(hi_9, hi_9);
                    float tv_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_169[0]));
                        (&tv_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_40[0]), "=f"(tv_40[1]), "=f"(tv_40[2]), "=f"(tv_40[3]), "=f"(tv_40[4]), "=f"(tv_40[5]), "=f"(tv_40[6]), "=f"(tv_40[7]) : "r"(r_169[0]));
                        #endif
                    }
                    float2 a_165 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_160 = make_float2(tv_40[0], tv_40[1]);
                    float2 r_171 = fma_f32x2_rn_noftz(q_160, scale0_9, a_165);
                    acc_4[0] = r_171.x;
                    acc_4[1] = r_171.y;
                    float2 a_166 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_161 = make_float2(tv_40[2], tv_40[3]);
                    float2 r_172 = fma_f32x2_rn_noftz(q_161, scale0_9, a_166);
                    acc_4[2] = r_172.x;
                    acc_4[3] = r_172.y;
                    float2 a_167 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_162 = make_float2(tv_40[4], tv_40[5]);
                    float2 r_173 = fma_f32x2_rn_noftz(q_162, scale0_9, a_167);
                    acc_4[4] = r_173.x;
                    acc_4[5] = r_173.y;
                    float2 a_168 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_163 = make_float2(tv_40[6], tv_40[7]);
                    float2 r_174 = fma_f32x2_rn_noftz(q_163, scale0_9, a_168);
                    acc_4[6] = r_174.x;
                    acc_4[7] = r_174.y;
                    float tv_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_169[1 + 0]));
                        (&tv_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_41[0]), "=f"(tv_41[1]), "=f"(tv_41[2]), "=f"(tv_41[3]), "=f"(tv_41[4]), "=f"(tv_41[5]), "=f"(tv_41[6]), "=f"(tv_41[7]) : "r"(r_169[1 + 0]));
                        #endif
                    }
                    float2 a_169 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_164 = make_float2(tv_41[0], tv_41[1]);
                    float2 r_175 = fma_f32x2_rn_noftz(q_164, scale0_9, a_169);
                    acc_4[8] = r_175.x;
                    acc_4[9] = r_175.y;
                    float2 a_170 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_165 = make_float2(tv_41[2], tv_41[3]);
                    float2 r_176 = fma_f32x2_rn_noftz(q_165, scale0_9, a_170);
                    acc_4[10] = r_176.x;
                    acc_4[11] = r_176.y;
                    float2 a_171 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_166 = make_float2(tv_41[4], tv_41[5]);
                    float2 r_177 = fma_f32x2_rn_noftz(q_166, scale0_9, a_171);
                    acc_4[12] = r_177.x;
                    acc_4[13] = r_177.y;
                    float2 a_172 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_167 = make_float2(tv_41[6], tv_41[7]);
                    float2 r_178 = fma_f32x2_rn_noftz(q_167, scale0_9, a_172);
                    acc_4[14] = r_178.x;
                    acc_4[15] = r_178.y;
                    float tv_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_169[2 + 0]));
                        (&tv_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_42[0]), "=f"(tv_42[1]), "=f"(tv_42[2]), "=f"(tv_42[3]), "=f"(tv_42[4]), "=f"(tv_42[5]), "=f"(tv_42[6]), "=f"(tv_42[7]) : "r"(r_169[2 + 0]));
                        #endif
                    }
                    float2 a_173 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_168 = make_float2(tv_42[0], tv_42[1]);
                    float2 r_179 = fma_f32x2_rn_noftz(q_168, scale1_9, a_173);
                    acc_4[16] = r_179.x;
                    acc_4[17] = r_179.y;
                    float2 a_174 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_169 = make_float2(tv_42[2], tv_42[3]);
                    float2 r_180 = fma_f32x2_rn_noftz(q_169, scale1_9, a_174);
                    acc_4[18] = r_180.x;
                    acc_4[19] = r_180.y;
                    float2 a_175 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_170 = make_float2(tv_42[4], tv_42[5]);
                    float2 r_181 = fma_f32x2_rn_noftz(q_170, scale1_9, a_175);
                    acc_4[20] = r_181.x;
                    acc_4[21] = r_181.y;
                    float2 a_176 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_171 = make_float2(tv_42[6], tv_42[7]);
                    float2 r_182 = fma_f32x2_rn_noftz(q_171, scale1_9, a_176);
                    acc_4[22] = r_182.x;
                    acc_4[23] = r_182.y;
                    float tv_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_169[3 + 0]));
                        (&tv_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_43[0]), "=f"(tv_43[1]), "=f"(tv_43[2]), "=f"(tv_43[3]), "=f"(tv_43[4]), "=f"(tv_43[5]), "=f"(tv_43[6]), "=f"(tv_43[7]) : "r"(r_169[3 + 0]));
                        #endif
                    }
                    float2 a_177 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_172 = make_float2(tv_43[0], tv_43[1]);
                    float2 r_183 = fma_f32x2_rn_noftz(q_172, scale1_9, a_177);
                    acc_4[24] = r_183.x;
                    acc_4[25] = r_183.y;
                    float2 a_178 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_173 = make_float2(tv_43[2], tv_43[3]);
                    float2 r_184 = fma_f32x2_rn_noftz(q_173, scale1_9, a_178);
                    acc_4[26] = r_184.x;
                    acc_4[27] = r_184.y;
                    float2 a_179 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_174 = make_float2(tv_43[4], tv_43[5]);
                    float2 r_185 = fma_f32x2_rn_noftz(q_174, scale1_9, a_179);
                    acc_4[28] = r_185.x;
                    acc_4[29] = r_185.y;
                    float2 a_180 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_175 = make_float2(tv_43[6], tv_43[7]);
                    float2 r_186 = fma_f32x2_rn_noftz(q_175, scale1_9, a_180);
                    acc_4[30] = r_186.x;
                    acc_4[31] = r_186.y;
                    uint32_t e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_10) : "h"((uint16_t)(b_sc_4)));
                    float cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_11) : "h"((uint16_t)(e4m3x2_to_f16x2_10)));
                    float cvt_f32_f16_1_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_10) : "h"((uint16_t)(e4m3x2_to_f16x2_10 >> 16)));
                    float s0_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_10) : "f"(cvt_f32_f16_11), "f"(rs_b_f_0));
                    float s1_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_10) : "f"(cvt_f32_f16_1_10), "f"(rs_b_f_0));
                    float lo_10 = s0_10;
                    float hi_10 = s1_10;
                    if (fhi_14) {
                        lo_10 = s1_10;
                        hi_10 = s0_10;
                    }
                    float2 scale0_10 = make_float2(lo_10, lo_10);
                    float2 scale1_10 = make_float2(hi_10, hi_10);
                    float tv_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_170[0]));
                        (&tv_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_44[0]), "=f"(tv_44[1]), "=f"(tv_44[2]), "=f"(tv_44[3]), "=f"(tv_44[4]), "=f"(tv_44[5]), "=f"(tv_44[6]), "=f"(tv_44[7]) : "r"(r_170[0]));
                        #endif
                    }
                    float2 a_181 = make_float2(acc_4[0], acc_4[1]);
                    float2 q_176 = make_float2(tv_44[0], tv_44[1]);
                    float2 r_187 = fma_f32x2_rn_noftz(q_176, scale0_10, a_181);
                    acc_4[0] = r_187.x;
                    acc_4[1] = r_187.y;
                    float2 a_182 = make_float2(acc_4[2], acc_4[3]);
                    float2 q_177 = make_float2(tv_44[2], tv_44[3]);
                    float2 r_188 = fma_f32x2_rn_noftz(q_177, scale0_10, a_182);
                    acc_4[2] = r_188.x;
                    acc_4[3] = r_188.y;
                    float2 a_183 = make_float2(acc_4[4], acc_4[5]);
                    float2 q_178 = make_float2(tv_44[4], tv_44[5]);
                    float2 r_189 = fma_f32x2_rn_noftz(q_178, scale0_10, a_183);
                    acc_4[4] = r_189.x;
                    acc_4[5] = r_189.y;
                    float2 a_184 = make_float2(acc_4[6], acc_4[7]);
                    float2 q_179 = make_float2(tv_44[6], tv_44[7]);
                    float2 r_190 = fma_f32x2_rn_noftz(q_179, scale0_10, a_184);
                    acc_4[6] = r_190.x;
                    acc_4[7] = r_190.y;
                    float tv_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_170[1 + 0]));
                        (&tv_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_45[0]), "=f"(tv_45[1]), "=f"(tv_45[2]), "=f"(tv_45[3]), "=f"(tv_45[4]), "=f"(tv_45[5]), "=f"(tv_45[6]), "=f"(tv_45[7]) : "r"(r_170[1 + 0]));
                        #endif
                    }
                    float2 a_185 = make_float2(acc_4[8], acc_4[9]);
                    float2 q_180 = make_float2(tv_45[0], tv_45[1]);
                    float2 r_191 = fma_f32x2_rn_noftz(q_180, scale0_10, a_185);
                    acc_4[8] = r_191.x;
                    acc_4[9] = r_191.y;
                    float2 a_186 = make_float2(acc_4[10], acc_4[11]);
                    float2 q_181 = make_float2(tv_45[2], tv_45[3]);
                    float2 r_192 = fma_f32x2_rn_noftz(q_181, scale0_10, a_186);
                    acc_4[10] = r_192.x;
                    acc_4[11] = r_192.y;
                    float2 a_187 = make_float2(acc_4[12], acc_4[13]);
                    float2 q_182 = make_float2(tv_45[4], tv_45[5]);
                    float2 r_193 = fma_f32x2_rn_noftz(q_182, scale0_10, a_187);
                    acc_4[12] = r_193.x;
                    acc_4[13] = r_193.y;
                    float2 a_188 = make_float2(acc_4[14], acc_4[15]);
                    float2 q_183 = make_float2(tv_45[6], tv_45[7]);
                    float2 r_194 = fma_f32x2_rn_noftz(q_183, scale0_10, a_188);
                    acc_4[14] = r_194.x;
                    acc_4[15] = r_194.y;
                    float tv_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_170[2 + 0]));
                        (&tv_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_46[0]), "=f"(tv_46[1]), "=f"(tv_46[2]), "=f"(tv_46[3]), "=f"(tv_46[4]), "=f"(tv_46[5]), "=f"(tv_46[6]), "=f"(tv_46[7]) : "r"(r_170[2 + 0]));
                        #endif
                    }
                    float2 a_189 = make_float2(acc_4[16], acc_4[17]);
                    float2 q_184 = make_float2(tv_46[0], tv_46[1]);
                    float2 r_195 = fma_f32x2_rn_noftz(q_184, scale1_10, a_189);
                    acc_4[16] = r_195.x;
                    acc_4[17] = r_195.y;
                    float2 a_190 = make_float2(acc_4[18], acc_4[19]);
                    float2 q_185 = make_float2(tv_46[2], tv_46[3]);
                    float2 r_196 = fma_f32x2_rn_noftz(q_185, scale1_10, a_190);
                    acc_4[18] = r_196.x;
                    acc_4[19] = r_196.y;
                    float2 a_191 = make_float2(acc_4[20], acc_4[21]);
                    float2 q_186 = make_float2(tv_46[4], tv_46[5]);
                    float2 r_197 = fma_f32x2_rn_noftz(q_186, scale1_10, a_191);
                    acc_4[20] = r_197.x;
                    acc_4[21] = r_197.y;
                    float2 a_192 = make_float2(acc_4[22], acc_4[23]);
                    float2 q_187 = make_float2(tv_46[6], tv_46[7]);
                    float2 r_198 = fma_f32x2_rn_noftz(q_187, scale1_10, a_192);
                    acc_4[22] = r_198.x;
                    acc_4[23] = r_198.y;
                    float tv_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_170[3 + 0]));
                        (&tv_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&tv_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&tv_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&tv_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&tv_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&tv_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&tv_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&tv_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(tv_47[0]), "=f"(tv_47[1]), "=f"(tv_47[2]), "=f"(tv_47[3]), "=f"(tv_47[4]), "=f"(tv_47[5]), "=f"(tv_47[6]), "=f"(tv_47[7]) : "r"(r_170[3 + 0]));
                        #endif
                    }
                    float2 a_193 = make_float2(acc_4[24], acc_4[25]);
                    float2 q_188 = make_float2(tv_47[0], tv_47[1]);
                    float2 r_199 = fma_f32x2_rn_noftz(q_188, scale1_10, a_193);
                    acc_4[24] = r_199.x;
                    acc_4[25] = r_199.y;
                    float2 a_194 = make_float2(acc_4[26], acc_4[27]);
                    float2 q_189 = make_float2(tv_47[2], tv_47[3]);
                    float2 r_200 = fma_f32x2_rn_noftz(q_189, scale1_10, a_194);
                    acc_4[26] = r_200.x;
                    acc_4[27] = r_200.y;
                    float2 a_195 = make_float2(acc_4[28], acc_4[29]);
                    float2 q_190 = make_float2(tv_47[4], tv_47[5]);
                    float2 r_201 = fma_f32x2_rn_noftz(q_190, scale1_10, a_195);
                    acc_4[28] = r_201.x;
                    acc_4[29] = r_201.y;
                    float2 a_196 = make_float2(acc_4[30], acc_4[31]);
                    float2 q_191 = make_float2(tv_47[6], tv_47[7]);
                    float2 r_202 = fma_f32x2_rn_noftz(q_191, scale1_10, a_196);
                    acc_4[30] = r_202.x;
                    acc_4[31] = r_202.y;
                    unsigned int t_4[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_4[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 3))));
                    float2 a2_1_80 = make_float2(acc_4[0], acc_4[1]);
                    float2 z_1_80 = add_f32x2_noftz(a2_1_80, make_float2(__uint_as_float(t_4[0] << 16), __uint_as_float(t_4[0] & 4294901760u)));
                    acc_4[0] = z_1_80.x;
                    acc_4[1] = z_1_80.y;
                    float2 ss2_1_80 = fma_f32x2_rn_noftz(z_1_80, z_1_80, make_float2(ssx, ssy));
                    ssx = ss2_1_80.x;
                    ssy = ss2_1_80.y;
                    float2 a2_1_81 = make_float2(acc_4[2], acc_4[3]);
                    float2 z_1_81 = add_f32x2_noftz(a2_1_81, make_float2(__uint_as_float(t_4[1] << 16), __uint_as_float(t_4[1] & 4294901760u)));
                    acc_4[2] = z_1_81.x;
                    acc_4[3] = z_1_81.y;
                    float2 ss2_1_81 = fma_f32x2_rn_noftz(z_1_81, z_1_81, make_float2(ssx, ssy));
                    ssx = ss2_1_81.x;
                    ssy = ss2_1_81.y;
                    float2 a2_1_82 = make_float2(acc_4[4], acc_4[5]);
                    float2 z_1_82 = add_f32x2_noftz(a2_1_82, make_float2(__uint_as_float(t_4[2] << 16), __uint_as_float(t_4[2] & 4294901760u)));
                    acc_4[4] = z_1_82.x;
                    acc_4[5] = z_1_82.y;
                    float2 ss2_1_82 = fma_f32x2_rn_noftz(z_1_82, z_1_82, make_float2(ssx, ssy));
                    ssx = ss2_1_82.x;
                    ssy = ss2_1_82.y;
                    float2 a2_1_83 = make_float2(acc_4[6], acc_4[7]);
                    float2 z_1_83 = add_f32x2_noftz(a2_1_83, make_float2(__uint_as_float(t_4[3] << 16), __uint_as_float(t_4[3] & 4294901760u)));
                    acc_4[6] = z_1_83.x;
                    acc_4[7] = z_1_83.y;
                    float2 ss2_1_83 = fma_f32x2_rn_noftz(z_1_83, z_1_83, make_float2(ssx, ssy));
                    ssx = ss2_1_83.x;
                    ssy = ss2_1_83.y;
                    float2 a2_1_84 = make_float2(acc_4[8], acc_4[9]);
                    float2 z_1_84 = add_f32x2_noftz(a2_1_84, make_float2(__uint_as_float(t_4[4] << 16), __uint_as_float(t_4[4] & 4294901760u)));
                    acc_4[8] = z_1_84.x;
                    acc_4[9] = z_1_84.y;
                    float2 ss2_1_84 = fma_f32x2_rn_noftz(z_1_84, z_1_84, make_float2(ssx, ssy));
                    ssx = ss2_1_84.x;
                    ssy = ss2_1_84.y;
                    float2 a2_1_85 = make_float2(acc_4[10], acc_4[11]);
                    float2 z_1_85 = add_f32x2_noftz(a2_1_85, make_float2(__uint_as_float(t_4[5] << 16), __uint_as_float(t_4[5] & 4294901760u)));
                    acc_4[10] = z_1_85.x;
                    acc_4[11] = z_1_85.y;
                    float2 ss2_1_85 = fma_f32x2_rn_noftz(z_1_85, z_1_85, make_float2(ssx, ssy));
                    ssx = ss2_1_85.x;
                    ssy = ss2_1_85.y;
                    float2 a2_1_86 = make_float2(acc_4[12], acc_4[13]);
                    float2 z_1_86 = add_f32x2_noftz(a2_1_86, make_float2(__uint_as_float(t_4[6] << 16), __uint_as_float(t_4[6] & 4294901760u)));
                    acc_4[12] = z_1_86.x;
                    acc_4[13] = z_1_86.y;
                    float2 ss2_1_86 = fma_f32x2_rn_noftz(z_1_86, z_1_86, make_float2(ssx, ssy));
                    ssx = ss2_1_86.x;
                    ssy = ss2_1_86.y;
                    float2 a2_1_87 = make_float2(acc_4[14], acc_4[15]);
                    float2 z_1_87 = add_f32x2_noftz(a2_1_87, make_float2(__uint_as_float(t_4[7] << 16), __uint_as_float(t_4[7] & 4294901760u)));
                    acc_4[14] = z_1_87.x;
                    acc_4[15] = z_1_87.y;
                    float2 ss2_1_87 = fma_f32x2_rn_noftz(z_1_87, z_1_87, make_float2(ssx, ssy));
                    ssx = ss2_1_87.x;
                    ssy = ss2_1_87.y;
                    float2 a2_1_88 = make_float2(acc_4[16], acc_4[17]);
                    float2 z_1_88 = add_f32x2_noftz(a2_1_88, make_float2(__uint_as_float(t_4[8] << 16), __uint_as_float(t_4[8] & 4294901760u)));
                    acc_4[16] = z_1_88.x;
                    acc_4[17] = z_1_88.y;
                    float2 ss2_1_88 = fma_f32x2_rn_noftz(z_1_88, z_1_88, make_float2(ssx, ssy));
                    ssx = ss2_1_88.x;
                    ssy = ss2_1_88.y;
                    float2 a2_1_89 = make_float2(acc_4[18], acc_4[19]);
                    float2 z_1_89 = add_f32x2_noftz(a2_1_89, make_float2(__uint_as_float(t_4[9] << 16), __uint_as_float(t_4[9] & 4294901760u)));
                    acc_4[18] = z_1_89.x;
                    acc_4[19] = z_1_89.y;
                    float2 ss2_1_89 = fma_f32x2_rn_noftz(z_1_89, z_1_89, make_float2(ssx, ssy));
                    ssx = ss2_1_89.x;
                    ssy = ss2_1_89.y;
                    float2 a2_1_90 = make_float2(acc_4[20], acc_4[21]);
                    float2 z_1_90 = add_f32x2_noftz(a2_1_90, make_float2(__uint_as_float(t_4[10] << 16), __uint_as_float(t_4[10] & 4294901760u)));
                    acc_4[20] = z_1_90.x;
                    acc_4[21] = z_1_90.y;
                    float2 ss2_1_90 = fma_f32x2_rn_noftz(z_1_90, z_1_90, make_float2(ssx, ssy));
                    ssx = ss2_1_90.x;
                    ssy = ss2_1_90.y;
                    float2 a2_1_91 = make_float2(acc_4[22], acc_4[23]);
                    float2 z_1_91 = add_f32x2_noftz(a2_1_91, make_float2(__uint_as_float(t_4[11] << 16), __uint_as_float(t_4[11] & 4294901760u)));
                    acc_4[22] = z_1_91.x;
                    acc_4[23] = z_1_91.y;
                    float2 ss2_1_91 = fma_f32x2_rn_noftz(z_1_91, z_1_91, make_float2(ssx, ssy));
                    ssx = ss2_1_91.x;
                    ssy = ss2_1_91.y;
                    float2 a2_1_92 = make_float2(acc_4[24], acc_4[25]);
                    float2 z_1_92 = add_f32x2_noftz(a2_1_92, make_float2(__uint_as_float(t_4[12] << 16), __uint_as_float(t_4[12] & 4294901760u)));
                    acc_4[24] = z_1_92.x;
                    acc_4[25] = z_1_92.y;
                    float2 ss2_1_92 = fma_f32x2_rn_noftz(z_1_92, z_1_92, make_float2(ssx, ssy));
                    ssx = ss2_1_92.x;
                    ssy = ss2_1_92.y;
                    float2 a2_1_93 = make_float2(acc_4[26], acc_4[27]);
                    float2 z_1_93 = add_f32x2_noftz(a2_1_93, make_float2(__uint_as_float(t_4[13] << 16), __uint_as_float(t_4[13] & 4294901760u)));
                    acc_4[26] = z_1_93.x;
                    acc_4[27] = z_1_93.y;
                    float2 ss2_1_93 = fma_f32x2_rn_noftz(z_1_93, z_1_93, make_float2(ssx, ssy));
                    ssx = ss2_1_93.x;
                    ssy = ss2_1_93.y;
                    float2 a2_1_94 = make_float2(acc_4[28], acc_4[29]);
                    float2 z_1_94 = add_f32x2_noftz(a2_1_94, make_float2(__uint_as_float(t_4[14] << 16), __uint_as_float(t_4[14] & 4294901760u)));
                    acc_4[28] = z_1_94.x;
                    acc_4[29] = z_1_94.y;
                    float2 ss2_1_94 = fma_f32x2_rn_noftz(z_1_94, z_1_94, make_float2(ssx, ssy));
                    ssx = ss2_1_94.x;
                    ssy = ss2_1_94.y;
                    float2 a2_1_95 = make_float2(acc_4[30], acc_4[31]);
                    float2 z_1_95 = add_f32x2_noftz(a2_1_95, make_float2(__uint_as_float(t_4[15] << 16), __uint_as_float(t_4[15] & 4294901760u)));
                    acc_4[30] = z_1_95.x;
                    acc_4[31] = z_1_95.y;
                    float2 ss2_1_95 = fma_f32x2_rn_noftz(z_1_95, z_1_95, make_float2(ssx, ssy));
                    ssx = ss2_1_95.x;
                    ssy = ss2_1_95.y;
                    unsigned int acc_bf16_4[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
                        acc_bf16_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 0))), "r"(acc_bf16_4[0]), "r"(acc_bf16_4[1]), "r"(acc_bf16_4[2]), "r"(acc_bf16_4[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 1))), "r"(acc_bf16_4[4]), "r"(acc_bf16_4[5]), "r"(acc_bf16_4[6]), "r"(acc_bf16_4[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 2))), "r"(acc_bf16_4[8]), "r"(acc_bf16_4[9]), "r"(acc_bf16_4[10]), "r"(acc_bf16_4[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 3))), "r"(acc_bf16_4[12]), "r"(acc_bf16_4[13]), "r"(acc_bf16_4[14]), "r"(acc_bf16_4[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        {
                            void* _cpbulk_dst_8 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(res_out) + (drow * 12288));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_8), "r"(wsm_addr + (unsigned int)rb_off), "r"((uint32_t)(12288))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    float ss;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(ss) : "f"(ssx), "f"(ssy));
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(ss), "f"(_shfl_xor_sync_10));
                    ss = _add_9;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                    float _add_11;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(ss), "f"(_shfl_xor_sync_12));
                    ss = _add_11;
                    float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                    float _add_13;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(ss), "f"(_shfl_xor_sync_14));
                    ss = _add_13;
                    float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                    float _add_15;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(ss), "f"(_shfl_xor_sync_16));
                    ss = _add_15;
                    float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                    float _add_17;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(ss), "f"(_shfl_xor_sync_18));
                    ss = _add_17;
                    float _fdiv_19 = __fdiv_rn(ss, 6144.0f);
                    float arg;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(arg) : "f"(_fdiv_19), "f"(eps));
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
                        int pp = lane_2 + 32 * j;
                        unsigned int tx[16];
                        unsigned int tw[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[0])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[4])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[4])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[8])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[8])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[12])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp + 16 * (f ^ 3))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[12])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp + 16 * (f ^ 3))));
                        float y[32];
                        float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(tx[0] << 16), __uint_as_float(tx[0] & 4294901760u));
                        float2 tz_0;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&r2));
                        float2 tw_0 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[0] << 16), __uint_as_float(tw[0] & 4294901760u)));
                        float2 t_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&tz_0), "l"(*(const unsigned long long*)&tw_0));
                        y[0] = t_5.x;
                        y[1] = t_5.y;
                        float2 _f32x2_mul_a_21 = make_float2(__uint_as_float(tx[1] << 16), __uint_as_float(tx[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&r2));
                        float2 tw_1 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[1] << 16), __uint_as_float(tw[1] & 4294901760u)));
                        float2 t_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_1));
                        y[2] = t_6.x;
                        y[3] = t_6.y;
                        float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(tx[2] << 16), __uint_as_float(tx[2] & 4294901760u));
                        float2 tz_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[2] << 16), __uint_as_float(tw[2] & 4294901760u)));
                        float2 t_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&tz_2), "l"(*(const unsigned long long*)&tw_2));
                        y[4] = t_7.x;
                        y[5] = t_7.y;
                        float2 _f32x2_mul_a_23 = make_float2(__uint_as_float(tx[3] << 16), __uint_as_float(tx[3] & 4294901760u));
                        float2 tz_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&r2));
                        float2 tw_3 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[3] << 16), __uint_as_float(tw[3] & 4294901760u)));
                        float2 t_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&tz_3), "l"(*(const unsigned long long*)&tw_3));
                        y[6] = t_8.x;
                        y[7] = t_8.y;
                        float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(tx[4] << 16), __uint_as_float(tx[4] & 4294901760u));
                        float2 tz_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&r2));
                        float2 tw_4 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[4] << 16), __uint_as_float(tw[4] & 4294901760u)));
                        float2 t_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&tz_4), "l"(*(const unsigned long long*)&tw_4));
                        y[8] = t_9.x;
                        y[9] = t_9.y;
                        float2 _f32x2_mul_a_25 = make_float2(__uint_as_float(tx[5] << 16), __uint_as_float(tx[5] & 4294901760u));
                        float2 tz_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&r2));
                        float2 tw_5 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[5] << 16), __uint_as_float(tw[5] & 4294901760u)));
                        float2 t_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&tz_5), "l"(*(const unsigned long long*)&tw_5));
                        y[10] = t_10.x;
                        y[11] = t_10.y;
                        float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(tx[6] << 16), __uint_as_float(tx[6] & 4294901760u));
                        float2 tz_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&r2));
                        float2 tw_6 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[6] << 16), __uint_as_float(tw[6] & 4294901760u)));
                        float2 t_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&tz_6), "l"(*(const unsigned long long*)&tw_6));
                        y[12] = t_11.x;
                        y[13] = t_11.y;
                        float2 _f32x2_mul_a_27 = make_float2(__uint_as_float(tx[7] << 16), __uint_as_float(tx[7] & 4294901760u));
                        float2 tz_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&r2));
                        float2 tw_7 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[7] << 16), __uint_as_float(tw[7] & 4294901760u)));
                        float2 t_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&tz_7), "l"(*(const unsigned long long*)&tw_7));
                        y[14] = t_12.x;
                        y[15] = t_12.y;
                        float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(tx[8] << 16), __uint_as_float(tx[8] & 4294901760u));
                        float2 tz_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&r2));
                        float2 tw_8 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[8] << 16), __uint_as_float(tw[8] & 4294901760u)));
                        float2 t_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&tz_8), "l"(*(const unsigned long long*)&tw_8));
                        y[16] = t_13.x;
                        y[17] = t_13.y;
                        float2 _f32x2_mul_a_29 = make_float2(__uint_as_float(tx[9] << 16), __uint_as_float(tx[9] & 4294901760u));
                        float2 tz_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&r2));
                        float2 tw_9 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[9] << 16), __uint_as_float(tw[9] & 4294901760u)));
                        float2 t_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&tz_9), "l"(*(const unsigned long long*)&tw_9));
                        y[18] = t_14.x;
                        y[19] = t_14.y;
                        float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(tx[10] << 16), __uint_as_float(tx[10] & 4294901760u));
                        float2 tz_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&r2));
                        float2 tw_10 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[10] << 16), __uint_as_float(tw[10] & 4294901760u)));
                        float2 t_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&tz_10), "l"(*(const unsigned long long*)&tw_10));
                        y[20] = t_15.x;
                        y[21] = t_15.y;
                        float2 _f32x2_mul_a_31 = make_float2(__uint_as_float(tx[11] << 16), __uint_as_float(tx[11] & 4294901760u));
                        float2 tz_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&r2));
                        float2 tw_11 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[11] << 16), __uint_as_float(tw[11] & 4294901760u)));
                        float2 t_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&tz_11), "l"(*(const unsigned long long*)&tw_11));
                        y[22] = t_16.x;
                        y[23] = t_16.y;
                        float2 _f32x2_mul_a_32 = make_float2(__uint_as_float(tx[12] << 16), __uint_as_float(tx[12] & 4294901760u));
                        float2 tz_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&r2));
                        float2 tw_12 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[12] << 16), __uint_as_float(tw[12] & 4294901760u)));
                        float2 t_17;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&tz_12), "l"(*(const unsigned long long*)&tw_12));
                        y[24] = t_17.x;
                        y[25] = t_17.y;
                        float2 _f32x2_mul_a_33 = make_float2(__uint_as_float(tx[13] << 16), __uint_as_float(tx[13] & 4294901760u));
                        float2 tz_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&r2));
                        float2 tw_13 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[13] << 16), __uint_as_float(tw[13] & 4294901760u)));
                        float2 t_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&tz_13), "l"(*(const unsigned long long*)&tw_13));
                        y[26] = t_18.x;
                        y[27] = t_18.y;
                        float2 _f32x2_mul_a_34 = make_float2(__uint_as_float(tx[14] << 16), __uint_as_float(tx[14] & 4294901760u));
                        float2 tz_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_34), "l"(*(const unsigned long long*)&r2));
                        float2 tw_14 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[14] << 16), __uint_as_float(tw[14] & 4294901760u)));
                        float2 t_19;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&tz_14), "l"(*(const unsigned long long*)&tw_14));
                        y[28] = t_19.x;
                        y[29] = t_19.y;
                        float2 _f32x2_mul_a_35 = make_float2(__uint_as_float(tx[15] << 16), __uint_as_float(tx[15] & 4294901760u));
                        float2 tz_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&r2));
                        float2 tw_15 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[15] << 16), __uint_as_float(tw[15] & 4294901760u)));
                        float2 t_20;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&tz_15), "l"(*(const unsigned long long*)&tw_15));
                        y[30] = t_20.x;
                        y[31] = t_20.y;
                        unsigned int y_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        long long hoff = sf_base + (long long)(intile + (unsigned int)(8192 * j));
                        float zero = 0.0f;
                        bool fodd_11 = (f & 1) != 0;
                        bool fhi_17 = (f & 2) != 0;
                        float bm[2];
                        unsigned int a_197 = 0;
                        uint32_t ab_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_0) : "r"(y_bf16[0]));
                        uint32_t mx_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_0) : "r"(a_197), "r"(ab_0));
                        a_197 = mx_0;
                        uint32_t ab_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_1) : "r"(y_bf16[1]));
                        uint32_t mx_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_1) : "r"(a_197), "r"(ab_1));
                        a_197 = mx_1;
                        uint32_t ab_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_2) : "r"(y_bf16[2]));
                        uint32_t mx_2;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_2) : "r"(a_197), "r"(ab_2));
                        a_197 = mx_2;
                        uint32_t ab_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_3) : "r"(y_bf16[3]));
                        uint32_t mx_3;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_3) : "r"(a_197), "r"(ab_3));
                        a_197 = mx_3;
                        uint32_t ab_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_4) : "r"(y_bf16[4]));
                        uint32_t mx_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_4) : "r"(a_197), "r"(ab_4));
                        a_197 = mx_4;
                        uint32_t ab_5;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_5) : "r"(y_bf16[5]));
                        uint32_t mx_5;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_5) : "r"(a_197), "r"(ab_5));
                        a_197 = mx_5;
                        uint32_t ab_6;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_6) : "r"(y_bf16[6]));
                        uint32_t mx_6;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_6) : "r"(a_197), "r"(ab_6));
                        a_197 = mx_6;
                        uint32_t ab_7;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_7) : "r"(y_bf16[7]));
                        uint32_t mx_7;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_7) : "r"(a_197), "r"(ab_7));
                        a_197 = mx_7;
                        bm[0] = fmaxf(__uint_as_float(a_197 << 16), __uint_as_float(a_197 & 4294901760u));
                        unsigned int a_198 = 0;
                        uint32_t ab_8;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_8) : "r"(y_bf16[8]));
                        uint32_t mx_8;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_8) : "r"(a_198), "r"(ab_8));
                        a_198 = mx_8;
                        uint32_t ab_9;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_9) : "r"(y_bf16[9]));
                        uint32_t mx_9;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_9) : "r"(a_198), "r"(ab_9));
                        a_198 = mx_9;
                        uint32_t ab_10;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_10) : "r"(y_bf16[10]));
                        uint32_t mx_10;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_10) : "r"(a_198), "r"(ab_10));
                        a_198 = mx_10;
                        uint32_t ab_11;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_11) : "r"(y_bf16[11]));
                        uint32_t mx_11;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_11) : "r"(a_198), "r"(ab_11));
                        a_198 = mx_11;
                        uint32_t ab_12;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_12) : "r"(y_bf16[12]));
                        uint32_t mx_12;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_12) : "r"(a_198), "r"(ab_12));
                        a_198 = mx_12;
                        uint32_t ab_13;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_13) : "r"(y_bf16[13]));
                        uint32_t mx_13;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_13) : "r"(a_198), "r"(ab_13));
                        a_198 = mx_13;
                        uint32_t ab_14;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_14) : "r"(y_bf16[14]));
                        uint32_t mx_14;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_14) : "r"(a_198), "r"(ab_14));
                        a_198 = mx_14;
                        uint32_t ab_15;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_15) : "r"(y_bf16[15]));
                        uint32_t mx_15;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_15) : "r"(a_198), "r"(ab_15));
                        a_198 = mx_15;
                        bm[1] = fmaxf(__uint_as_float(a_198 << 16), __uint_as_float(a_198 & 4294901760u));
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
                        if (fhi_17) {
                            vm0 = bm[1];
                            vm1 = bm[0];
                        }
                        unsigned int sfb[2];
                        float sfv;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv) : "f"(gso), "f"(mul_f32_rn_noftz(vm0, rcp6)));
                        uint16_t _e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(zero), "f"(sfv));
                        sfb[0] = (unsigned int)_e4m3x2_f32_6 & 255;
                        uint32_t e4m3x2_to_f16x2_11;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_11) : "h"((uint16_t)(_e4m3x2_f32_6)));
                        float cvt_f32_f16_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_12) : "h"((uint16_t)(e4m3x2_to_f16x2_11)));
                        float cvt_f32_f16_1_11;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_11) : "h"((uint16_t)(e4m3x2_to_f16x2_11 >> 16)));
                        float _rcp_0 = approx_rcp(mul_f32_rn_noftz(cvt_f32_f16_12, rcp_gs));
                        float oscale = 0.0f;
                        if (vm0 != 0.0f) {
                            oscale = _rcp_0;
                        }
                        float sfv_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv_47) : "f"(gso), "f"(mul_f32_rn_noftz(vm1, rcp6)));
                        uint16_t _e4m3x2_f32_7;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(zero), "f"(sfv_47));
                        sfb[1] = (unsigned int)_e4m3x2_f32_7 & 255;
                        uint32_t e4m3x2_to_f16x2_12;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_12) : "h"((uint16_t)(_e4m3x2_f32_7)));
                        float cvt_f32_f16_13;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_13) : "h"((uint16_t)(e4m3x2_to_f16x2_12)));
                        float cvt_f32_f16_1_12;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_12) : "h"((uint16_t)(e4m3x2_to_f16x2_12 >> 16)));
                        float _rcp_1 = approx_rcp(mul_f32_rn_noftz(cvt_f32_f16_13, rcp_gs));
                        float oscale_48 = 0.0f;
                        if (vm1 != 0.0f) {
                            oscale_48 = _rcp_1;
                        }
                        float os_p0 = oscale;
                        float os_p1 = oscale_48;
                        if (fhi_17) {
                            os_p0 = oscale_48;
                            os_p1 = oscale;
                        }
                        float2 os2_p0 = make_float2(os_p0, os_p0);
                        float2 os2_p1 = make_float2(os_p1, os_p1);
                        unsigned int w_49[4];
                        float2 _f32x2_mul_a_36 = make_float2(yb[0], yb[1]);
                        float2 g;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&_f32x2_mul_a_36), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_72;
                        uint16_t _e2m1_pair_h_37;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g.x), "f"(g.y));
                        _fp4_pair_72 = (uint32_t)_e2m1_pair_h_37;
                        float2 _f32x2_mul_a_38 = make_float2(yb[2], yb[3]);
                        float2 g_50;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_73;
                        uint16_t _e2m1_pair_h_39;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_50.x), "f"(g_50.y));
                        _fp4_pair_73 = (uint32_t)_e2m1_pair_h_39;
                        float2 _f32x2_mul_a_40 = make_float2(yb[4], yb[5]);
                        float2 g_51;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_74;
                        uint16_t _e2m1_pair_h_41;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_51.x), "f"(g_51.y));
                        _fp4_pair_74 = (uint32_t)_e2m1_pair_h_41;
                        float2 _f32x2_mul_a_42 = make_float2(yb[6], yb[7]);
                        float2 g_52;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_e2m1_pack_pair_43;
                        uint16_t _e2m1_pair_h_44;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_52.x), "f"(g_52.y));
                        _fp4_e2m1_pack_pair_43 = (uint32_t)_e2m1_pair_h_44;
                        w_49[0] = _fp4_pair_72 | _fp4_pair_73 << 8 | _fp4_pair_74 << 16 | _fp4_e2m1_pack_pair_43 << 24;
                        float2 _f32x2_mul_a_45 = make_float2(yb[8], yb[9]);
                        float2 g_53;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_75;
                        uint16_t _e2m1_pair_h_46;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_53.x), "f"(g_53.y));
                        _fp4_pair_75 = (uint32_t)_e2m1_pair_h_46;
                        float2 _f32x2_mul_a_47 = make_float2(yb[10], yb[11]);
                        float2 g_54;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_76;
                        uint16_t _e2m1_pair_h_48;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(g_54.x), "f"(g_54.y));
                        _fp4_pair_76 = (uint32_t)_e2m1_pair_h_48;
                        float2 _f32x2_mul_a_49 = make_float2(yb[12], yb[13]);
                        float2 g_55;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_77;
                        uint16_t _e2m1_pair_h_50;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_50) : "f"(g_55.x), "f"(g_55.y));
                        _fp4_pair_77 = (uint32_t)_e2m1_pair_h_50;
                        float2 _f32x2_mul_a_51 = make_float2(yb[14], yb[15]);
                        float2 g_56;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_e2m1_pack_pair_52;
                        uint16_t _e2m1_pair_h_53;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_53) : "f"(g_56.x), "f"(g_56.y));
                        _fp4_e2m1_pack_pair_52 = (uint32_t)_e2m1_pair_h_53;
                        w_49[1] = _fp4_pair_75 | _fp4_pair_76 << 8 | _fp4_pair_77 << 16 | _fp4_e2m1_pack_pair_52 << 24;
                        float2 _f32x2_mul_a_54 = make_float2(yb[16], yb[17]);
                        float2 g_57;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_78;
                        uint16_t _e2m1_pair_h_55;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_57.x), "f"(g_57.y));
                        _fp4_pair_78 = (uint32_t)_e2m1_pair_h_55;
                        float2 _f32x2_mul_a_56 = make_float2(yb[18], yb[19]);
                        float2 g_58;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_79;
                        uint16_t _e2m1_pair_h_57;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_57) : "f"(g_58.x), "f"(g_58.y));
                        _fp4_pair_79 = (uint32_t)_e2m1_pair_h_57;
                        float2 _f32x2_mul_a_58 = make_float2(yb[20], yb[21]);
                        float2 g_59;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_80;
                        uint16_t _e2m1_pair_h_59;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_59.x), "f"(g_59.y));
                        _fp4_pair_80 = (uint32_t)_e2m1_pair_h_59;
                        float2 _f32x2_mul_a_60 = make_float2(yb[22], yb[23]);
                        float2 g_60;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_e2m1_pack_pair_61;
                        uint16_t _e2m1_pair_h_62;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_62) : "f"(g_60.x), "f"(g_60.y));
                        _fp4_e2m1_pack_pair_61 = (uint32_t)_e2m1_pair_h_62;
                        w_49[2] = _fp4_pair_78 | _fp4_pair_79 << 8 | _fp4_pair_80 << 16 | _fp4_e2m1_pack_pair_61 << 24;
                        float2 _f32x2_mul_a_63 = make_float2(yb[24], yb[25]);
                        float2 g_61;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_81;
                        uint16_t _e2m1_pair_h_64;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_61.x), "f"(g_61.y));
                        _fp4_pair_81 = (uint32_t)_e2m1_pair_h_64;
                        float2 _f32x2_mul_a_65 = make_float2(yb[26], yb[27]);
                        float2 g_62;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_62) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_82;
                        uint16_t _e2m1_pair_h_66;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_62.x), "f"(g_62.y));
                        _fp4_pair_82 = (uint32_t)_e2m1_pair_h_66;
                        float2 _f32x2_mul_a_67 = make_float2(yb[28], yb[29]);
                        float2 g_63;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_83;
                        uint16_t _e2m1_pair_h_68;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_63.x), "f"(g_63.y));
                        _fp4_pair_83 = (uint32_t)_e2m1_pair_h_68;
                        float2 _f32x2_mul_a_69 = make_float2(yb[30], yb[31]);
                        float2 g_64;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_64) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_e2m1_pack_pair_70;
                        uint16_t _e2m1_pair_h_71;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_71) : "f"(g_64.x), "f"(g_64.y));
                        _fp4_e2m1_pack_pair_70 = (uint32_t)_e2m1_pair_h_71;
                        w_49[3] = _fp4_pair_81 | _fp4_pair_82 << 8 | _fp4_pair_83 << 16 | _fp4_e2m1_pack_pair_70 << 24;
                        unsigned int sw[4];
                        sw[0] = w_49[0];
                        sw[1] = w_49[1];
                        sw[2] = w_49[2];
                        sw[3] = w_49[3];
                        if (fodd_11) {
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
                        if (fhi_17) {
                            words[0] = sw[2];
                            words[1] = sw[3];
                            words[2] = sw[0];
                            words[3] = sw[1];
                        }
                        long long vidx = drow * 3072 + (long long)(16 * pp) >> 2;
                        {
                            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(outq) + vidx) = _iv4;
                        }
                        unsigned int half = sfb[0] | sfb[1] << 8;
                        unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, half, 1);
                        unsigned int oddmask = -(unsigned int)(lane_2 & 1);
                        unsigned int lo_11 = half & ~oddmask | _shfl_xor_0 & oddmask;
                        unsigned int hi_11 = _shfl_xor_0 & ~oddmask | half & oddmask;
                        unsigned int word_0 = lo_11 | hi_11 << 16;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out_sf + hoff)) + (0)) = word_0;
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
                long long lo_12 = u * bid64_1 / g64_1;
                long long hi_12 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_12 + lane64;
                #pragma unroll 1
                for (long long g_1 = g0; g_1 < hi_12; g_1 += 32) {
                    long long widx = v_base_1 + 16 * g_1 >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_12_1 = u_0 * bid64_1 / g64_1;
                long long hi_12_1 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_0 = lo_12_1 + lane64;
                #pragma unroll 1
                for (long long g_2 = g0_0; g_2 < hi_12_1; g_2 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_2 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx_1) = _iv4;
                    }
                }
                long long u_1 = pr64 * 16 / 16;
                long long lo_13 = u_1 * bid64_1 / g64_1;
                long long hi_13 = u_1 * (bid64_1 + 1) / g64_1;
                long long g0_1 = lo_13 + lane64;
                #pragma unroll 1
                for (long long g_3 = g0_1; g_3 < hi_13; g_3 += 32) {
                    long long widx_2 = m_base_1 + 16 * g_3 >> 2;
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
