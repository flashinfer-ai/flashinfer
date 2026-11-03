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
#define SMEM_WSM_STAGE_BYTES 211552
#define SMEM_WSM_STRIDE 211552
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 211552
#define SMEM_WSM32_STRIDE 211552
#define SMEM_TOTAL 225024
#define THREADS 384
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp2_rmsnorm_bf16_nvfp4_p2d_w11_qp_sfu_csp_sop(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ outq, uint8_t* __restrict__ out_sf, float* __restrict__ gs_out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
    #define rb_full_addr (mbar_base + 88)

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

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // xb_full: 11 barriers, init_count=1
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
            // rb_full: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: workers ----
    if (warp <= 10) {
        { // workers_main
            int tid_0 = (int)tid;
            #pragma unroll 1
            for (int i = tid_0; i < 768; i += 352) {
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
                n = (nrows - warp_1 + 11 - 1) / 11;
            }
            if (n > 0 && lane_2 == 0) {
                long long first_row = (long long)(r0 + warp_1);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 12288);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 19232), x + (first_row * 6144), 12288, xb_full_addr + (warp_1) * 8);
            }
            if (tid == 0) {
                int base = bid * 4;
                ctl[0] = ctrl[base];
                ctl[1] = ctrl[base + 2];
                ctl[2] = ctrl[base + 3];
            }
            asm volatile("barrier.sync 0, 384;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int call = ctl[0];
            unsigned int buf = call % 3;
            int f = lane_2 >> 1 & 3;
            int xb_off = warp_1 * 19232;
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
            unsigned int _phase_xb_full = 0;
            unsigned int _phase_rb_full = 1;
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 11 * k);
                    int ob_off = own_off + k % 2 * 3472;
                    if (k > 1) {
                        if (lane_2 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        __syncwarp();
                    }
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (warp_1) * 8, _phase_xb_full);
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
                    int pp_r_0 = lane_2 + 32;
                    int base_r_0 = xb_off + 64 * pp_r_0;
                    unsigned int a_r_0 = 0;
                    unsigned int b_r_0 = 0;
                    unsigned int c0_r_0[4];
                    unsigned int c1_r_0[4];
                    unsigned int c2_r_0[4];
                    unsigned int c3_r_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c0_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_0 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_0 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_0 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_r_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_0 + 16 * (f ^ 3))));
                    uint32_t _bf16x2_abs_16;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_16) : "r"(c0_r_0[0]));
                    uint32_t _bf16x2_max_16;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_16) : "r"(a_r_0), "r"(_bf16x2_abs_16));
                    a_r_0 = _bf16x2_max_16;
                    uint32_t _bf16x2_abs_17;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_17) : "r"(c0_r_0[1]));
                    uint32_t _bf16x2_max_17;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_17) : "r"(a_r_0), "r"(_bf16x2_abs_17));
                    a_r_0 = _bf16x2_max_17;
                    uint32_t _bf16x2_abs_18;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_18) : "r"(c0_r_0[2]));
                    uint32_t _bf16x2_max_18;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_18) : "r"(a_r_0), "r"(_bf16x2_abs_18));
                    a_r_0 = _bf16x2_max_18;
                    uint32_t _bf16x2_abs_19;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_19) : "r"(c0_r_0[3]));
                    uint32_t _bf16x2_max_19;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_19) : "r"(a_r_0), "r"(_bf16x2_abs_19));
                    a_r_0 = _bf16x2_max_19;
                    uint32_t _bf16x2_abs_20;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_20) : "r"(c1_r_0[0]));
                    uint32_t _bf16x2_max_20;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_20) : "r"(a_r_0), "r"(_bf16x2_abs_20));
                    a_r_0 = _bf16x2_max_20;
                    uint32_t _bf16x2_abs_21;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_21) : "r"(c1_r_0[1]));
                    uint32_t _bf16x2_max_21;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_21) : "r"(a_r_0), "r"(_bf16x2_abs_21));
                    a_r_0 = _bf16x2_max_21;
                    uint32_t _bf16x2_abs_22;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_22) : "r"(c1_r_0[2]));
                    uint32_t _bf16x2_max_22;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_22) : "r"(a_r_0), "r"(_bf16x2_abs_22));
                    a_r_0 = _bf16x2_max_22;
                    uint32_t _bf16x2_abs_23;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_23) : "r"(c1_r_0[3]));
                    uint32_t _bf16x2_max_23;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_23) : "r"(a_r_0), "r"(_bf16x2_abs_23));
                    a_r_0 = _bf16x2_max_23;
                    uint32_t _bf16x2_abs_24;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_24) : "r"(c2_r_0[0]));
                    uint32_t _bf16x2_max_24;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_24) : "r"(b_r_0), "r"(_bf16x2_abs_24));
                    b_r_0 = _bf16x2_max_24;
                    uint32_t _bf16x2_abs_25;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_25) : "r"(c2_r_0[1]));
                    uint32_t _bf16x2_max_25;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_25) : "r"(b_r_0), "r"(_bf16x2_abs_25));
                    b_r_0 = _bf16x2_max_25;
                    uint32_t _bf16x2_abs_26;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_26) : "r"(c2_r_0[2]));
                    uint32_t _bf16x2_max_26;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_26) : "r"(b_r_0), "r"(_bf16x2_abs_26));
                    b_r_0 = _bf16x2_max_26;
                    uint32_t _bf16x2_abs_27;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_27) : "r"(c2_r_0[3]));
                    uint32_t _bf16x2_max_27;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_27) : "r"(b_r_0), "r"(_bf16x2_abs_27));
                    b_r_0 = _bf16x2_max_27;
                    uint32_t _bf16x2_abs_28;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_28) : "r"(c3_r_0[0]));
                    uint32_t _bf16x2_max_28;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_28) : "r"(b_r_0), "r"(_bf16x2_abs_28));
                    b_r_0 = _bf16x2_max_28;
                    uint32_t _bf16x2_abs_29;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_29) : "r"(c3_r_0[1]));
                    uint32_t _bf16x2_max_29;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_29) : "r"(b_r_0), "r"(_bf16x2_abs_29));
                    b_r_0 = _bf16x2_max_29;
                    uint32_t _bf16x2_abs_30;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_30) : "r"(c3_r_0[2]));
                    uint32_t _bf16x2_max_30;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_30) : "r"(b_r_0), "r"(_bf16x2_abs_30));
                    b_r_0 = _bf16x2_max_30;
                    uint32_t _bf16x2_abs_31;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_31) : "r"(c3_r_0[3]));
                    uint32_t _bf16x2_max_31;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_31) : "r"(b_r_0), "r"(_bf16x2_abs_31));
                    b_r_0 = _bf16x2_max_31;
                    unsigned int xa_r_0 = ((a_r_0 & 65535) > (a_r_0 >> 16) ? (a_r_0 & 65535) : (a_r_0 >> 16));
                    unsigned int xb_r_0 = ((b_r_0 & 65535) > (b_r_0 >> 16) ? (b_r_0 & 65535) : (b_r_0 >> 16));
                    if ((f & 2) != 0) {
                        bmp[1] = xb_r_0 | xa_r_0 << 16;
                    } else {
                        bmp[1] = xa_r_0 | xb_r_0 << 16;
                    }
                    unsigned int _max_2 = ((xa_r_0) > (xb_r_0) ? (xa_r_0) : (xb_r_0));
                    am = fmaxf(am, __uint_as_float(_max_2 << 16));
                    int pp_r_1 = lane_2 + 64;
                    int base_r_1 = xb_off + 64 * pp_r_1;
                    unsigned int a_r_1 = 0;
                    unsigned int b_r_1 = 0;
                    unsigned int c0_r_1[4];
                    unsigned int c1_r_1[4];
                    unsigned int c2_r_1[4];
                    unsigned int c3_r_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c0_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_1 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_1 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_1 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_r_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_1 + 16 * (f ^ 3))));
                    uint32_t _bf16x2_abs_32;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_32) : "r"(c0_r_1[0]));
                    uint32_t _bf16x2_max_32;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_32) : "r"(a_r_1), "r"(_bf16x2_abs_32));
                    a_r_1 = _bf16x2_max_32;
                    uint32_t _bf16x2_abs_33;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_33) : "r"(c0_r_1[1]));
                    uint32_t _bf16x2_max_33;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_33) : "r"(a_r_1), "r"(_bf16x2_abs_33));
                    a_r_1 = _bf16x2_max_33;
                    uint32_t _bf16x2_abs_34;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_34) : "r"(c0_r_1[2]));
                    uint32_t _bf16x2_max_34;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_34) : "r"(a_r_1), "r"(_bf16x2_abs_34));
                    a_r_1 = _bf16x2_max_34;
                    uint32_t _bf16x2_abs_35;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_35) : "r"(c0_r_1[3]));
                    uint32_t _bf16x2_max_35;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_35) : "r"(a_r_1), "r"(_bf16x2_abs_35));
                    a_r_1 = _bf16x2_max_35;
                    uint32_t _bf16x2_abs_36;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_36) : "r"(c1_r_1[0]));
                    uint32_t _bf16x2_max_36;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_36) : "r"(a_r_1), "r"(_bf16x2_abs_36));
                    a_r_1 = _bf16x2_max_36;
                    uint32_t _bf16x2_abs_37;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_37) : "r"(c1_r_1[1]));
                    uint32_t _bf16x2_max_37;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_37) : "r"(a_r_1), "r"(_bf16x2_abs_37));
                    a_r_1 = _bf16x2_max_37;
                    uint32_t _bf16x2_abs_38;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_38) : "r"(c1_r_1[2]));
                    uint32_t _bf16x2_max_38;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_38) : "r"(a_r_1), "r"(_bf16x2_abs_38));
                    a_r_1 = _bf16x2_max_38;
                    uint32_t _bf16x2_abs_39;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_39) : "r"(c1_r_1[3]));
                    uint32_t _bf16x2_max_39;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_39) : "r"(a_r_1), "r"(_bf16x2_abs_39));
                    a_r_1 = _bf16x2_max_39;
                    uint32_t _bf16x2_abs_40;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_40) : "r"(c2_r_1[0]));
                    uint32_t _bf16x2_max_40;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_40) : "r"(b_r_1), "r"(_bf16x2_abs_40));
                    b_r_1 = _bf16x2_max_40;
                    uint32_t _bf16x2_abs_41;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_41) : "r"(c2_r_1[1]));
                    uint32_t _bf16x2_max_41;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_41) : "r"(b_r_1), "r"(_bf16x2_abs_41));
                    b_r_1 = _bf16x2_max_41;
                    uint32_t _bf16x2_abs_42;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_42) : "r"(c2_r_1[2]));
                    uint32_t _bf16x2_max_42;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_42) : "r"(b_r_1), "r"(_bf16x2_abs_42));
                    b_r_1 = _bf16x2_max_42;
                    uint32_t _bf16x2_abs_43;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_43) : "r"(c2_r_1[3]));
                    uint32_t _bf16x2_max_43;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_43) : "r"(b_r_1), "r"(_bf16x2_abs_43));
                    b_r_1 = _bf16x2_max_43;
                    uint32_t _bf16x2_abs_44;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_44) : "r"(c3_r_1[0]));
                    uint32_t _bf16x2_max_44;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_44) : "r"(b_r_1), "r"(_bf16x2_abs_44));
                    b_r_1 = _bf16x2_max_44;
                    uint32_t _bf16x2_abs_45;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_45) : "r"(c3_r_1[1]));
                    uint32_t _bf16x2_max_45;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_45) : "r"(b_r_1), "r"(_bf16x2_abs_45));
                    b_r_1 = _bf16x2_max_45;
                    uint32_t _bf16x2_abs_46;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_46) : "r"(c3_r_1[2]));
                    uint32_t _bf16x2_max_46;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_46) : "r"(b_r_1), "r"(_bf16x2_abs_46));
                    b_r_1 = _bf16x2_max_46;
                    uint32_t _bf16x2_abs_47;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_47) : "r"(c3_r_1[3]));
                    uint32_t _bf16x2_max_47;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_47) : "r"(b_r_1), "r"(_bf16x2_abs_47));
                    b_r_1 = _bf16x2_max_47;
                    unsigned int xa_r_1 = ((a_r_1 & 65535) > (a_r_1 >> 16) ? (a_r_1 & 65535) : (a_r_1 >> 16));
                    unsigned int xb_r_1 = ((b_r_1 & 65535) > (b_r_1 >> 16) ? (b_r_1 & 65535) : (b_r_1 >> 16));
                    if ((f & 2) != 0) {
                        bmp[2] = xb_r_1 | xa_r_1 << 16;
                    } else {
                        bmp[2] = xa_r_1 | xb_r_1 << 16;
                    }
                    unsigned int _max_3 = ((xa_r_1) > (xb_r_1) ? (xa_r_1) : (xb_r_1));
                    am = fmaxf(am, __uint_as_float(_max_3 << 16));
                    int pp_r_2 = lane_2 + 96;
                    int base_r_2 = xb_off + 64 * pp_r_2;
                    unsigned int a_r_2 = 0;
                    unsigned int b_r_2 = 0;
                    unsigned int c0_r_2[4];
                    unsigned int c1_r_2[4];
                    unsigned int c2_r_2[4];
                    unsigned int c3_r_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c0_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_2 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_2 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_2 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_r_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_2 + 16 * (f ^ 3))));
                    uint32_t _bf16x2_abs_48;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_48) : "r"(c0_r_2[0]));
                    uint32_t _bf16x2_max_48;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_48) : "r"(a_r_2), "r"(_bf16x2_abs_48));
                    a_r_2 = _bf16x2_max_48;
                    uint32_t _bf16x2_abs_49;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_49) : "r"(c0_r_2[1]));
                    uint32_t _bf16x2_max_49;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_49) : "r"(a_r_2), "r"(_bf16x2_abs_49));
                    a_r_2 = _bf16x2_max_49;
                    uint32_t _bf16x2_abs_50;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_50) : "r"(c0_r_2[2]));
                    uint32_t _bf16x2_max_50;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_50) : "r"(a_r_2), "r"(_bf16x2_abs_50));
                    a_r_2 = _bf16x2_max_50;
                    uint32_t _bf16x2_abs_51;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_51) : "r"(c0_r_2[3]));
                    uint32_t _bf16x2_max_51;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_51) : "r"(a_r_2), "r"(_bf16x2_abs_51));
                    a_r_2 = _bf16x2_max_51;
                    uint32_t _bf16x2_abs_52;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_52) : "r"(c1_r_2[0]));
                    uint32_t _bf16x2_max_52;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_52) : "r"(a_r_2), "r"(_bf16x2_abs_52));
                    a_r_2 = _bf16x2_max_52;
                    uint32_t _bf16x2_abs_53;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_53) : "r"(c1_r_2[1]));
                    uint32_t _bf16x2_max_53;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_53) : "r"(a_r_2), "r"(_bf16x2_abs_53));
                    a_r_2 = _bf16x2_max_53;
                    uint32_t _bf16x2_abs_54;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_54) : "r"(c1_r_2[2]));
                    uint32_t _bf16x2_max_54;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_54) : "r"(a_r_2), "r"(_bf16x2_abs_54));
                    a_r_2 = _bf16x2_max_54;
                    uint32_t _bf16x2_abs_55;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_55) : "r"(c1_r_2[3]));
                    uint32_t _bf16x2_max_55;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_55) : "r"(a_r_2), "r"(_bf16x2_abs_55));
                    a_r_2 = _bf16x2_max_55;
                    uint32_t _bf16x2_abs_56;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_56) : "r"(c2_r_2[0]));
                    uint32_t _bf16x2_max_56;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_56) : "r"(b_r_2), "r"(_bf16x2_abs_56));
                    b_r_2 = _bf16x2_max_56;
                    uint32_t _bf16x2_abs_57;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_57) : "r"(c2_r_2[1]));
                    uint32_t _bf16x2_max_57;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_57) : "r"(b_r_2), "r"(_bf16x2_abs_57));
                    b_r_2 = _bf16x2_max_57;
                    uint32_t _bf16x2_abs_58;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_58) : "r"(c2_r_2[2]));
                    uint32_t _bf16x2_max_58;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_58) : "r"(b_r_2), "r"(_bf16x2_abs_58));
                    b_r_2 = _bf16x2_max_58;
                    uint32_t _bf16x2_abs_59;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_59) : "r"(c2_r_2[3]));
                    uint32_t _bf16x2_max_59;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_59) : "r"(b_r_2), "r"(_bf16x2_abs_59));
                    b_r_2 = _bf16x2_max_59;
                    uint32_t _bf16x2_abs_60;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_60) : "r"(c3_r_2[0]));
                    uint32_t _bf16x2_max_60;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_60) : "r"(b_r_2), "r"(_bf16x2_abs_60));
                    b_r_2 = _bf16x2_max_60;
                    uint32_t _bf16x2_abs_61;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_61) : "r"(c3_r_2[1]));
                    uint32_t _bf16x2_max_61;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_61) : "r"(b_r_2), "r"(_bf16x2_abs_61));
                    b_r_2 = _bf16x2_max_61;
                    uint32_t _bf16x2_abs_62;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_62) : "r"(c3_r_2[2]));
                    uint32_t _bf16x2_max_62;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_62) : "r"(b_r_2), "r"(_bf16x2_abs_62));
                    b_r_2 = _bf16x2_max_62;
                    uint32_t _bf16x2_abs_63;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_63) : "r"(c3_r_2[3]));
                    uint32_t _bf16x2_max_63;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_63) : "r"(b_r_2), "r"(_bf16x2_abs_63));
                    b_r_2 = _bf16x2_max_63;
                    unsigned int xa_r_2 = ((a_r_2 & 65535) > (a_r_2 >> 16) ? (a_r_2 & 65535) : (a_r_2 >> 16));
                    unsigned int xb_r_2 = ((b_r_2 & 65535) > (b_r_2 >> 16) ? (b_r_2 & 65535) : (b_r_2 >> 16));
                    if ((f & 2) != 0) {
                        bmp[3] = xb_r_2 | xa_r_2 << 16;
                    } else {
                        bmp[3] = xa_r_2 | xb_r_2 << 16;
                    }
                    unsigned int _max_4 = ((xa_r_2) > (xb_r_2) ? (xa_r_2) : (xb_r_2));
                    am = fmaxf(am, __uint_as_float(_max_4 << 16));
                    int pp_r_3 = lane_2 + 128;
                    int base_r_3 = xb_off + 64 * pp_r_3;
                    unsigned int a_r_3 = 0;
                    unsigned int b_r_3 = 0;
                    unsigned int c0_r_3[4];
                    unsigned int c1_r_3[4];
                    unsigned int c2_r_3[4];
                    unsigned int c3_r_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c0_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_3 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_3 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_3 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_r_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_3 + 16 * (f ^ 3))));
                    uint32_t _bf16x2_abs_64;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_64) : "r"(c0_r_3[0]));
                    uint32_t _bf16x2_max_64;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_64) : "r"(a_r_3), "r"(_bf16x2_abs_64));
                    a_r_3 = _bf16x2_max_64;
                    uint32_t _bf16x2_abs_65;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_65) : "r"(c0_r_3[1]));
                    uint32_t _bf16x2_max_65;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_65) : "r"(a_r_3), "r"(_bf16x2_abs_65));
                    a_r_3 = _bf16x2_max_65;
                    uint32_t _bf16x2_abs_66;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_66) : "r"(c0_r_3[2]));
                    uint32_t _bf16x2_max_66;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_66) : "r"(a_r_3), "r"(_bf16x2_abs_66));
                    a_r_3 = _bf16x2_max_66;
                    uint32_t _bf16x2_abs_67;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_67) : "r"(c0_r_3[3]));
                    uint32_t _bf16x2_max_67;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_67) : "r"(a_r_3), "r"(_bf16x2_abs_67));
                    a_r_3 = _bf16x2_max_67;
                    uint32_t _bf16x2_abs_68;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_68) : "r"(c1_r_3[0]));
                    uint32_t _bf16x2_max_68;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_68) : "r"(a_r_3), "r"(_bf16x2_abs_68));
                    a_r_3 = _bf16x2_max_68;
                    uint32_t _bf16x2_abs_69;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_69) : "r"(c1_r_3[1]));
                    uint32_t _bf16x2_max_69;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_69) : "r"(a_r_3), "r"(_bf16x2_abs_69));
                    a_r_3 = _bf16x2_max_69;
                    uint32_t _bf16x2_abs_70;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_70) : "r"(c1_r_3[2]));
                    uint32_t _bf16x2_max_70;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_70) : "r"(a_r_3), "r"(_bf16x2_abs_70));
                    a_r_3 = _bf16x2_max_70;
                    uint32_t _bf16x2_abs_71;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_71) : "r"(c1_r_3[3]));
                    uint32_t _bf16x2_max_71;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_71) : "r"(a_r_3), "r"(_bf16x2_abs_71));
                    a_r_3 = _bf16x2_max_71;
                    uint32_t _bf16x2_abs_72;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_72) : "r"(c2_r_3[0]));
                    uint32_t _bf16x2_max_72;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_72) : "r"(b_r_3), "r"(_bf16x2_abs_72));
                    b_r_3 = _bf16x2_max_72;
                    uint32_t _bf16x2_abs_73;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_73) : "r"(c2_r_3[1]));
                    uint32_t _bf16x2_max_73;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_73) : "r"(b_r_3), "r"(_bf16x2_abs_73));
                    b_r_3 = _bf16x2_max_73;
                    uint32_t _bf16x2_abs_74;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_74) : "r"(c2_r_3[2]));
                    uint32_t _bf16x2_max_74;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_74) : "r"(b_r_3), "r"(_bf16x2_abs_74));
                    b_r_3 = _bf16x2_max_74;
                    uint32_t _bf16x2_abs_75;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_75) : "r"(c2_r_3[3]));
                    uint32_t _bf16x2_max_75;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_75) : "r"(b_r_3), "r"(_bf16x2_abs_75));
                    b_r_3 = _bf16x2_max_75;
                    uint32_t _bf16x2_abs_76;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_76) : "r"(c3_r_3[0]));
                    uint32_t _bf16x2_max_76;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_76) : "r"(b_r_3), "r"(_bf16x2_abs_76));
                    b_r_3 = _bf16x2_max_76;
                    uint32_t _bf16x2_abs_77;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_77) : "r"(c3_r_3[1]));
                    uint32_t _bf16x2_max_77;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_77) : "r"(b_r_3), "r"(_bf16x2_abs_77));
                    b_r_3 = _bf16x2_max_77;
                    uint32_t _bf16x2_abs_78;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_78) : "r"(c3_r_3[2]));
                    uint32_t _bf16x2_max_78;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_78) : "r"(b_r_3), "r"(_bf16x2_abs_78));
                    b_r_3 = _bf16x2_max_78;
                    uint32_t _bf16x2_abs_79;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_79) : "r"(c3_r_3[3]));
                    uint32_t _bf16x2_max_79;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_79) : "r"(b_r_3), "r"(_bf16x2_abs_79));
                    b_r_3 = _bf16x2_max_79;
                    unsigned int xa_r_3 = ((a_r_3 & 65535) > (a_r_3 >> 16) ? (a_r_3 & 65535) : (a_r_3 >> 16));
                    unsigned int xb_r_3 = ((b_r_3 & 65535) > (b_r_3 >> 16) ? (b_r_3 & 65535) : (b_r_3 >> 16));
                    if ((f & 2) != 0) {
                        bmp[4] = xb_r_3 | xa_r_3 << 16;
                    } else {
                        bmp[4] = xa_r_3 | xb_r_3 << 16;
                    }
                    unsigned int _max_5 = ((xa_r_3) > (xb_r_3) ? (xa_r_3) : (xb_r_3));
                    am = fmaxf(am, __uint_as_float(_max_5 << 16));
                    int pp_r_4 = lane_2 + 160;
                    int base_r_4 = xb_off + 64 * pp_r_4;
                    unsigned int a_r_4 = 0;
                    unsigned int b_r_4 = 0;
                    unsigned int c0_r_4[4];
                    unsigned int c1_r_4[4];
                    unsigned int c2_r_4[4];
                    unsigned int c3_r_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c0_r_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c0_r_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_4 + 16 * f)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c1_r_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c1_r_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_4 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c2_r_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c2_r_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_4 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&c3_r_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&c3_r_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base_r_4 + 16 * (f ^ 3))));
                    uint32_t _bf16x2_abs_80;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_80) : "r"(c0_r_4[0]));
                    uint32_t _bf16x2_max_80;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_80) : "r"(a_r_4), "r"(_bf16x2_abs_80));
                    a_r_4 = _bf16x2_max_80;
                    uint32_t _bf16x2_abs_81;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_81) : "r"(c0_r_4[1]));
                    uint32_t _bf16x2_max_81;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_81) : "r"(a_r_4), "r"(_bf16x2_abs_81));
                    a_r_4 = _bf16x2_max_81;
                    uint32_t _bf16x2_abs_82;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_82) : "r"(c0_r_4[2]));
                    uint32_t _bf16x2_max_82;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_82) : "r"(a_r_4), "r"(_bf16x2_abs_82));
                    a_r_4 = _bf16x2_max_82;
                    uint32_t _bf16x2_abs_83;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_83) : "r"(c0_r_4[3]));
                    uint32_t _bf16x2_max_83;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_83) : "r"(a_r_4), "r"(_bf16x2_abs_83));
                    a_r_4 = _bf16x2_max_83;
                    uint32_t _bf16x2_abs_84;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_84) : "r"(c1_r_4[0]));
                    uint32_t _bf16x2_max_84;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_84) : "r"(a_r_4), "r"(_bf16x2_abs_84));
                    a_r_4 = _bf16x2_max_84;
                    uint32_t _bf16x2_abs_85;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_85) : "r"(c1_r_4[1]));
                    uint32_t _bf16x2_max_85;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_85) : "r"(a_r_4), "r"(_bf16x2_abs_85));
                    a_r_4 = _bf16x2_max_85;
                    uint32_t _bf16x2_abs_86;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_86) : "r"(c1_r_4[2]));
                    uint32_t _bf16x2_max_86;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_86) : "r"(a_r_4), "r"(_bf16x2_abs_86));
                    a_r_4 = _bf16x2_max_86;
                    uint32_t _bf16x2_abs_87;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_87) : "r"(c1_r_4[3]));
                    uint32_t _bf16x2_max_87;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_87) : "r"(a_r_4), "r"(_bf16x2_abs_87));
                    a_r_4 = _bf16x2_max_87;
                    uint32_t _bf16x2_abs_88;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_88) : "r"(c2_r_4[0]));
                    uint32_t _bf16x2_max_88;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_88) : "r"(b_r_4), "r"(_bf16x2_abs_88));
                    b_r_4 = _bf16x2_max_88;
                    uint32_t _bf16x2_abs_89;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_89) : "r"(c2_r_4[1]));
                    uint32_t _bf16x2_max_89;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_89) : "r"(b_r_4), "r"(_bf16x2_abs_89));
                    b_r_4 = _bf16x2_max_89;
                    uint32_t _bf16x2_abs_90;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_90) : "r"(c2_r_4[2]));
                    uint32_t _bf16x2_max_90;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_90) : "r"(b_r_4), "r"(_bf16x2_abs_90));
                    b_r_4 = _bf16x2_max_90;
                    uint32_t _bf16x2_abs_91;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_91) : "r"(c2_r_4[3]));
                    uint32_t _bf16x2_max_91;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_91) : "r"(b_r_4), "r"(_bf16x2_abs_91));
                    b_r_4 = _bf16x2_max_91;
                    uint32_t _bf16x2_abs_92;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_92) : "r"(c3_r_4[0]));
                    uint32_t _bf16x2_max_92;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_92) : "r"(b_r_4), "r"(_bf16x2_abs_92));
                    b_r_4 = _bf16x2_max_92;
                    uint32_t _bf16x2_abs_93;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_93) : "r"(c3_r_4[1]));
                    uint32_t _bf16x2_max_93;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_93) : "r"(b_r_4), "r"(_bf16x2_abs_93));
                    b_r_4 = _bf16x2_max_93;
                    uint32_t _bf16x2_abs_94;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_94) : "r"(c3_r_4[2]));
                    uint32_t _bf16x2_max_94;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_94) : "r"(b_r_4), "r"(_bf16x2_abs_94));
                    b_r_4 = _bf16x2_max_94;
                    uint32_t _bf16x2_abs_95;
                    asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_95) : "r"(c3_r_4[3]));
                    uint32_t _bf16x2_max_95;
                    asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_95) : "r"(b_r_4), "r"(_bf16x2_abs_95));
                    b_r_4 = _bf16x2_max_95;
                    unsigned int xa_r_4 = ((a_r_4 & 65535) > (a_r_4 >> 16) ? (a_r_4 & 65535) : (a_r_4 >> 16));
                    unsigned int xb_r_4 = ((b_r_4 & 65535) > (b_r_4 >> 16) ? (b_r_4 & 65535) : (b_r_4 >> 16));
                    if ((f & 2) != 0) {
                        bmp[5] = xb_r_4 | xa_r_4 << 16;
                    } else {
                        bmp[5] = xa_r_4 | xb_r_4 << 16;
                    }
                    unsigned int _max_6 = ((xa_r_4) > (xb_r_4) ? (xa_r_4) : (xb_r_4));
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
                    int pp2_1_0 = lane_2;
                    int base2_1_0 = xb_off + 64 * pp2_1_0;
                    float bm0_1_0 = __uint_as_float(bmp[0] << 16);
                    float bm1_1_0 = __uint_as_float(bmp[0] & 4294901760u);
                    float _fdiv_8 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_0, inv), 6.0f);
                    float _fdiv_9 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_0, inv), 6.0f);
                    uint16_t e4m3x2_f32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32) : "f"(_fdiv_8), "f"(_fdiv_9));
                    uint32_t e4m3x2_to_f16x2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2) : "h"((uint16_t)(e4m3x2_f32)));
                    float cvt_f32_f16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16) : "h"((uint16_t)(e4m3x2_to_f16x2)));
                    float cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1) : "h"((uint16_t)(e4m3x2_to_f16x2 >> 16)));
                    float fdiv_rn = __fdiv_rn(inv, cvt_f32_f16);
                    float fdiv_rn_1 = __fdiv_rn(inv, cvt_f32_f16_1);
                    float m0 = 0.0f;
                    float m1 = 0.0f;
                    if (cvt_f32_f16 > 0.0f) {
                        m0 = fdiv_rn;
                    }
                    if (cvt_f32_f16_1 > 0.0f) {
                        m1 = fdiv_rn_1;
                    }
                    unsigned int w_0[4];
                    int cf_1_0 = f ^ 0;
                    unsigned int v4_1_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_0[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_0)));
                    float m_1_0 = m0;
                    if ((cf_1_0 & 2) != 0) {
                        m_1_0 = m1;
                    }
                    float2 mm = make_float2(m_1_0, m_1_0);
                    float2 pair = make_float2(__uint_as_float(v4_1_0[0] << 16), __uint_as_float(v4_1_0[0] & 4294901760u));
                    float2 g;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g) : "l"(*(const unsigned long long*)&pair), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair;
                    uint16_t _e2m1_pair_h_10;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(g.x), "f"(g.y));
                    fp4_pair = (uint32_t)_e2m1_pair_h_10;
                    float2 pair_1 = make_float2(__uint_as_float(v4_1_0[1] << 16), __uint_as_float(v4_1_0[1] & 4294901760u));
                    float2 g_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1) : "l"(*(const unsigned long long*)&pair_1), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair_1;
                    uint16_t _e2m1_pair_h_11;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(g_1.x), "f"(g_1.y));
                    fp4_pair_1 = (uint32_t)_e2m1_pair_h_11;
                    float2 pair_2 = make_float2(__uint_as_float(v4_1_0[2] << 16), __uint_as_float(v4_1_0[2] & 4294901760u));
                    float2 g_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2) : "l"(*(const unsigned long long*)&pair_2), "l"(*(const unsigned long long*)&mm));
                    uint32_t fp4_pair_2;
                    uint16_t _e2m1_pair_h_12;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(g_2.x), "f"(g_2.y));
                    fp4_pair_2 = (uint32_t)_e2m1_pair_h_12;
                    float2 pair_3 = make_float2(__uint_as_float(v4_1_0[3] << 16), __uint_as_float(v4_1_0[3] & 4294901760u));
                    float2 g_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3) : "l"(*(const unsigned long long*)&pair_3), "l"(*(const unsigned long long*)&mm));
                    uint32_t _fp4_e2m1_pack_pair_13;
                    uint16_t _e2m1_pair_h_14;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(g_3.x), "f"(g_3.y));
                    _fp4_e2m1_pack_pair_13 = (uint32_t)_e2m1_pair_h_14;
                    unsigned int word = fp4_pair | fp4_pair_1 << 8 | fp4_pair_2 << 16 | _fp4_e2m1_pack_pair_13 << 24;
                    unsigned int carry = (word & 2004318071) + 2004318071;
                    w_0[0] = word & (carry | 2004318071);
                    int cf_1_1 = f ^ 1;
                    unsigned int v4_1_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_1[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_1)));
                    float m_1_1 = m0;
                    if ((cf_1_1 & 2) != 0) {
                        m_1_1 = m1;
                    }
                    float2 mm_0 = make_float2(m_1_1, m_1_1);
                    float2 pair_0 = make_float2(__uint_as_float(v4_1_1[0] << 16), __uint_as_float(v4_1_1[0] & 4294901760u));
                    float2 g_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_0) : "l"(*(const unsigned long long*)&pair_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_0;
                    uint16_t _e2m1_pair_h_15;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(g_0.x), "f"(g_0.y));
                    fp4_pair_0 = (uint32_t)_e2m1_pair_h_15;
                    float2 pair_1_0 = make_float2(__uint_as_float(v4_1_1[1] << 16), __uint_as_float(v4_1_1[1] & 4294901760u));
                    float2 g_1_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_0) : "l"(*(const unsigned long long*)&pair_1_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_1_0;
                    uint16_t _e2m1_pair_h_16;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(g_1_0.x), "f"(g_1_0.y));
                    fp4_pair_1_0 = (uint32_t)_e2m1_pair_h_16;
                    float2 pair_2_0 = make_float2(__uint_as_float(v4_1_1[2] << 16), __uint_as_float(v4_1_1[2] & 4294901760u));
                    float2 g_2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_0) : "l"(*(const unsigned long long*)&pair_2_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t fp4_pair_2_0;
                    uint16_t _e2m1_pair_h_17;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(g_2_0.x), "f"(g_2_0.y));
                    fp4_pair_2_0 = (uint32_t)_e2m1_pair_h_17;
                    float2 pair_3_0 = make_float2(__uint_as_float(v4_1_1[3] << 16), __uint_as_float(v4_1_1[3] & 4294901760u));
                    float2 g_3_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_0) : "l"(*(const unsigned long long*)&pair_3_0), "l"(*(const unsigned long long*)&mm_0));
                    uint32_t _fp4_e2m1_pack_pair_18;
                    uint16_t _e2m1_pair_h_19;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(g_3_0.x), "f"(g_3_0.y));
                    _fp4_e2m1_pack_pair_18 = (uint32_t)_e2m1_pair_h_19;
                    unsigned int word_0 = fp4_pair_0 | fp4_pair_1_0 << 8 | fp4_pair_2_0 << 16 | _fp4_e2m1_pack_pair_18 << 24;
                    unsigned int carry_0 = (word_0 & 2004318071) + 2004318071;
                    w_0[1] = word_0 & (carry_0 | 2004318071);
                    int cf_1_2 = f ^ 2;
                    unsigned int v4_1_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_2[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_2)));
                    float m_1_2 = m0;
                    if ((cf_1_2 & 2) != 0) {
                        m_1_2 = m1;
                    }
                    float2 mm_1 = make_float2(m_1_2, m_1_2);
                    float2 pair_4 = make_float2(__uint_as_float(v4_1_2[0] << 16), __uint_as_float(v4_1_2[0] & 4294901760u));
                    float2 g_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_4) : "l"(*(const unsigned long long*)&pair_4), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_3;
                    uint16_t _e2m1_pair_h_20;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(g_4.x), "f"(g_4.y));
                    fp4_pair_3 = (uint32_t)_e2m1_pair_h_20;
                    float2 pair_1_1 = make_float2(__uint_as_float(v4_1_2[1] << 16), __uint_as_float(v4_1_2[1] & 4294901760u));
                    float2 g_1_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_1) : "l"(*(const unsigned long long*)&pair_1_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_1_1;
                    uint16_t _e2m1_pair_h_21;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(g_1_1.x), "f"(g_1_1.y));
                    fp4_pair_1_1 = (uint32_t)_e2m1_pair_h_21;
                    float2 pair_2_1 = make_float2(__uint_as_float(v4_1_2[2] << 16), __uint_as_float(v4_1_2[2] & 4294901760u));
                    float2 g_2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_1) : "l"(*(const unsigned long long*)&pair_2_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t fp4_pair_2_1;
                    uint16_t _e2m1_pair_h_22;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(g_2_1.x), "f"(g_2_1.y));
                    fp4_pair_2_1 = (uint32_t)_e2m1_pair_h_22;
                    float2 pair_3_1 = make_float2(__uint_as_float(v4_1_2[3] << 16), __uint_as_float(v4_1_2[3] & 4294901760u));
                    float2 g_3_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_1) : "l"(*(const unsigned long long*)&pair_3_1), "l"(*(const unsigned long long*)&mm_1));
                    uint32_t _fp4_e2m1_pack_pair_23;
                    uint16_t _e2m1_pair_h_24;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(g_3_1.x), "f"(g_3_1.y));
                    _fp4_e2m1_pack_pair_23 = (uint32_t)_e2m1_pair_h_24;
                    unsigned int word_1 = fp4_pair_3 | fp4_pair_1_1 << 8 | fp4_pair_2_1 << 16 | _fp4_e2m1_pack_pair_23 << 24;
                    unsigned int carry_1 = (word_1 & 2004318071) + 2004318071;
                    w_0[2] = word_1 & (carry_1 | 2004318071);
                    int cf_1_3 = f ^ 3;
                    unsigned int v4_1_3[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_3[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_0 + 16 * cf_1_3)));
                    float m_1_3 = m0;
                    if ((cf_1_3 & 2) != 0) {
                        m_1_3 = m1;
                    }
                    float2 mm_2 = make_float2(m_1_3, m_1_3);
                    float2 pair_5 = make_float2(__uint_as_float(v4_1_3[0] << 16), __uint_as_float(v4_1_3[0] & 4294901760u));
                    float2 g_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_5) : "l"(*(const unsigned long long*)&pair_5), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_4;
                    uint16_t _e2m1_pair_h_25;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_25) : "f"(g_5.x), "f"(g_5.y));
                    fp4_pair_4 = (uint32_t)_e2m1_pair_h_25;
                    float2 pair_1_2 = make_float2(__uint_as_float(v4_1_3[1] << 16), __uint_as_float(v4_1_3[1] & 4294901760u));
                    float2 g_1_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_2) : "l"(*(const unsigned long long*)&pair_1_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_1_2;
                    uint16_t _e2m1_pair_h_26;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_26) : "f"(g_1_2.x), "f"(g_1_2.y));
                    fp4_pair_1_2 = (uint32_t)_e2m1_pair_h_26;
                    float2 pair_2_2 = make_float2(__uint_as_float(v4_1_3[2] << 16), __uint_as_float(v4_1_3[2] & 4294901760u));
                    float2 g_2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_2) : "l"(*(const unsigned long long*)&pair_2_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t fp4_pair_2_2;
                    uint16_t _e2m1_pair_h_27;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_27) : "f"(g_2_2.x), "f"(g_2_2.y));
                    fp4_pair_2_2 = (uint32_t)_e2m1_pair_h_27;
                    float2 pair_3_2 = make_float2(__uint_as_float(v4_1_3[3] << 16), __uint_as_float(v4_1_3[3] & 4294901760u));
                    float2 g_3_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_2) : "l"(*(const unsigned long long*)&pair_3_2), "l"(*(const unsigned long long*)&mm_2));
                    uint32_t _fp4_e2m1_pack_pair_28;
                    uint16_t _e2m1_pair_h_29;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_29) : "f"(g_3_2.x), "f"(g_3_2.y));
                    _fp4_e2m1_pack_pair_28 = (uint32_t)_e2m1_pair_h_29;
                    unsigned int word_2 = fp4_pair_4 | fp4_pair_1_2 << 8 | fp4_pair_2_2 << 16 | _fp4_e2m1_pack_pair_28 << 24;
                    unsigned int carry_2 = (word_2 & 2004318071) + 2004318071;
                    w_0[3] = word_2 & (carry_2 | 2004318071);
                    bool fodd_1_0 = (f & 1) != 0;
                    bool fhi_1_0 = (f & 2) != 0;
                    unsigned int t0_1_0 = w_0[0];
                    unsigned int t1_1_0 = w_0[1];
                    unsigned int t2_1_0 = w_0[2];
                    unsigned int t3_1_0 = w_0[3];
                    if (fodd_1_0) {
                        t0_1_0 = w_0[1];
                        t1_1_0 = w_0[0];
                        t2_1_0 = w_0[3];
                        t3_1_0 = w_0[2];
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
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_0)), "h"((uint16_t)(e4m3x2_f32 & 32639)) : "memory");
                    int pp2_1_1 = lane_2 + 32;
                    int base2_1_1 = xb_off + 64 * pp2_1_1;
                    float bm0_1_1 = __uint_as_float(bmp[1] << 16);
                    float bm1_1_1 = __uint_as_float(bmp[1] & 4294901760u);
                    float _fdiv_30 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_1, inv), 6.0f);
                    float _fdiv_31 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_1, inv), 6.0f);
                    uint16_t e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_0) : "f"(_fdiv_30), "f"(_fdiv_31));
                    uint32_t e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_0) : "h"((uint16_t)(e4m3x2_f32_0)));
                    float cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0)));
                    float cvt_f32_f16_1_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_0) : "h"((uint16_t)(e4m3x2_to_f16x2_0 >> 16)));
                    float fdiv_rn_0 = __fdiv_rn(inv, cvt_f32_f16_0);
                    float fdiv_rn_1_0 = __fdiv_rn(inv, cvt_f32_f16_1_0);
                    float m0_0 = 0.0f;
                    float m1_0 = 0.0f;
                    if (cvt_f32_f16_0 > 0.0f) {
                        m0_0 = fdiv_rn_0;
                    }
                    if (cvt_f32_f16_1_0 > 0.0f) {
                        m1_0 = fdiv_rn_1_0;
                    }
                    unsigned int w_1[4];
                    int cf_1_4 = f ^ 0;
                    unsigned int v4_1_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_4[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_4)));
                    float m_1_4 = m0_0;
                    if ((cf_1_4 & 2) != 0) {
                        m_1_4 = m1_0;
                    }
                    float2 mm_3 = make_float2(m_1_4, m_1_4);
                    float2 pair_6 = make_float2(__uint_as_float(v4_1_4[0] << 16), __uint_as_float(v4_1_4[0] & 4294901760u));
                    float2 g_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_6) : "l"(*(const unsigned long long*)&pair_6), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_5;
                    uint16_t _e2m1_pair_h_32;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_32) : "f"(g_6.x), "f"(g_6.y));
                    fp4_pair_5 = (uint32_t)_e2m1_pair_h_32;
                    float2 pair_1_3 = make_float2(__uint_as_float(v4_1_4[1] << 16), __uint_as_float(v4_1_4[1] & 4294901760u));
                    float2 g_1_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_3) : "l"(*(const unsigned long long*)&pair_1_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_1_3;
                    uint16_t _e2m1_pair_h_33;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_33) : "f"(g_1_3.x), "f"(g_1_3.y));
                    fp4_pair_1_3 = (uint32_t)_e2m1_pair_h_33;
                    float2 pair_2_3 = make_float2(__uint_as_float(v4_1_4[2] << 16), __uint_as_float(v4_1_4[2] & 4294901760u));
                    float2 g_2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_3) : "l"(*(const unsigned long long*)&pair_2_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t fp4_pair_2_3;
                    uint16_t _e2m1_pair_h_34;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_34) : "f"(g_2_3.x), "f"(g_2_3.y));
                    fp4_pair_2_3 = (uint32_t)_e2m1_pair_h_34;
                    float2 pair_3_3 = make_float2(__uint_as_float(v4_1_4[3] << 16), __uint_as_float(v4_1_4[3] & 4294901760u));
                    float2 g_3_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_3) : "l"(*(const unsigned long long*)&pair_3_3), "l"(*(const unsigned long long*)&mm_3));
                    uint32_t _fp4_e2m1_pack_pair_35;
                    uint16_t _e2m1_pair_h_36;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_36) : "f"(g_3_3.x), "f"(g_3_3.y));
                    _fp4_e2m1_pack_pair_35 = (uint32_t)_e2m1_pair_h_36;
                    unsigned int word_3 = fp4_pair_5 | fp4_pair_1_3 << 8 | fp4_pair_2_3 << 16 | _fp4_e2m1_pack_pair_35 << 24;
                    unsigned int carry_3 = (word_3 & 2004318071) + 2004318071;
                    w_1[0] = word_3 & (carry_3 | 2004318071);
                    int cf_1_5 = f ^ 1;
                    unsigned int v4_1_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_5[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_5)));
                    float m_1_5 = m0_0;
                    if ((cf_1_5 & 2) != 0) {
                        m_1_5 = m1_0;
                    }
                    float2 mm_4 = make_float2(m_1_5, m_1_5);
                    float2 pair_7 = make_float2(__uint_as_float(v4_1_5[0] << 16), __uint_as_float(v4_1_5[0] & 4294901760u));
                    float2 g_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_7) : "l"(*(const unsigned long long*)&pair_7), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_6;
                    uint16_t _e2m1_pair_h_37;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_37) : "f"(g_7.x), "f"(g_7.y));
                    fp4_pair_6 = (uint32_t)_e2m1_pair_h_37;
                    float2 pair_1_4 = make_float2(__uint_as_float(v4_1_5[1] << 16), __uint_as_float(v4_1_5[1] & 4294901760u));
                    float2 g_1_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_4) : "l"(*(const unsigned long long*)&pair_1_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_1_4;
                    uint16_t _e2m1_pair_h_38;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_38) : "f"(g_1_4.x), "f"(g_1_4.y));
                    fp4_pair_1_4 = (uint32_t)_e2m1_pair_h_38;
                    float2 pair_2_4 = make_float2(__uint_as_float(v4_1_5[2] << 16), __uint_as_float(v4_1_5[2] & 4294901760u));
                    float2 g_2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_4) : "l"(*(const unsigned long long*)&pair_2_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t fp4_pair_2_4;
                    uint16_t _e2m1_pair_h_39;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_39) : "f"(g_2_4.x), "f"(g_2_4.y));
                    fp4_pair_2_4 = (uint32_t)_e2m1_pair_h_39;
                    float2 pair_3_4 = make_float2(__uint_as_float(v4_1_5[3] << 16), __uint_as_float(v4_1_5[3] & 4294901760u));
                    float2 g_3_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_4) : "l"(*(const unsigned long long*)&pair_3_4), "l"(*(const unsigned long long*)&mm_4));
                    uint32_t _fp4_e2m1_pack_pair_40;
                    uint16_t _e2m1_pair_h_41;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_41) : "f"(g_3_4.x), "f"(g_3_4.y));
                    _fp4_e2m1_pack_pair_40 = (uint32_t)_e2m1_pair_h_41;
                    unsigned int word_4 = fp4_pair_6 | fp4_pair_1_4 << 8 | fp4_pair_2_4 << 16 | _fp4_e2m1_pack_pair_40 << 24;
                    unsigned int carry_4 = (word_4 & 2004318071) + 2004318071;
                    w_1[1] = word_4 & (carry_4 | 2004318071);
                    int cf_1_6 = f ^ 2;
                    unsigned int v4_1_6[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_6[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_6)));
                    float m_1_6 = m0_0;
                    if ((cf_1_6 & 2) != 0) {
                        m_1_6 = m1_0;
                    }
                    float2 mm_5 = make_float2(m_1_6, m_1_6);
                    float2 pair_8 = make_float2(__uint_as_float(v4_1_6[0] << 16), __uint_as_float(v4_1_6[0] & 4294901760u));
                    float2 g_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_8) : "l"(*(const unsigned long long*)&pair_8), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_7;
                    uint16_t _e2m1_pair_h_42;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_42) : "f"(g_8.x), "f"(g_8.y));
                    fp4_pair_7 = (uint32_t)_e2m1_pair_h_42;
                    float2 pair_1_5 = make_float2(__uint_as_float(v4_1_6[1] << 16), __uint_as_float(v4_1_6[1] & 4294901760u));
                    float2 g_1_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_5) : "l"(*(const unsigned long long*)&pair_1_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_1_5;
                    uint16_t _e2m1_pair_h_43;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_43) : "f"(g_1_5.x), "f"(g_1_5.y));
                    fp4_pair_1_5 = (uint32_t)_e2m1_pair_h_43;
                    float2 pair_2_5 = make_float2(__uint_as_float(v4_1_6[2] << 16), __uint_as_float(v4_1_6[2] & 4294901760u));
                    float2 g_2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_5) : "l"(*(const unsigned long long*)&pair_2_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t fp4_pair_2_5;
                    uint16_t _e2m1_pair_h_44;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_44) : "f"(g_2_5.x), "f"(g_2_5.y));
                    fp4_pair_2_5 = (uint32_t)_e2m1_pair_h_44;
                    float2 pair_3_5 = make_float2(__uint_as_float(v4_1_6[3] << 16), __uint_as_float(v4_1_6[3] & 4294901760u));
                    float2 g_3_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_5) : "l"(*(const unsigned long long*)&pair_3_5), "l"(*(const unsigned long long*)&mm_5));
                    uint32_t _fp4_e2m1_pack_pair_45;
                    uint16_t _e2m1_pair_h_46;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_46) : "f"(g_3_5.x), "f"(g_3_5.y));
                    _fp4_e2m1_pack_pair_45 = (uint32_t)_e2m1_pair_h_46;
                    unsigned int word_5 = fp4_pair_7 | fp4_pair_1_5 << 8 | fp4_pair_2_5 << 16 | _fp4_e2m1_pack_pair_45 << 24;
                    unsigned int carry_5 = (word_5 & 2004318071) + 2004318071;
                    w_1[2] = word_5 & (carry_5 | 2004318071);
                    int cf_1_7 = f ^ 3;
                    unsigned int v4_1_7[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_7[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_1 + 16 * cf_1_7)));
                    float m_1_7 = m0_0;
                    if ((cf_1_7 & 2) != 0) {
                        m_1_7 = m1_0;
                    }
                    float2 mm_6 = make_float2(m_1_7, m_1_7);
                    float2 pair_9 = make_float2(__uint_as_float(v4_1_7[0] << 16), __uint_as_float(v4_1_7[0] & 4294901760u));
                    float2 g_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_9) : "l"(*(const unsigned long long*)&pair_9), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_8;
                    uint16_t _e2m1_pair_h_47;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_47) : "f"(g_9.x), "f"(g_9.y));
                    fp4_pair_8 = (uint32_t)_e2m1_pair_h_47;
                    float2 pair_1_6 = make_float2(__uint_as_float(v4_1_7[1] << 16), __uint_as_float(v4_1_7[1] & 4294901760u));
                    float2 g_1_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_6) : "l"(*(const unsigned long long*)&pair_1_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_1_6;
                    uint16_t _e2m1_pair_h_48;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_48) : "f"(g_1_6.x), "f"(g_1_6.y));
                    fp4_pair_1_6 = (uint32_t)_e2m1_pair_h_48;
                    float2 pair_2_6 = make_float2(__uint_as_float(v4_1_7[2] << 16), __uint_as_float(v4_1_7[2] & 4294901760u));
                    float2 g_2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_6) : "l"(*(const unsigned long long*)&pair_2_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t fp4_pair_2_6;
                    uint16_t _e2m1_pair_h_49;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_49) : "f"(g_2_6.x), "f"(g_2_6.y));
                    fp4_pair_2_6 = (uint32_t)_e2m1_pair_h_49;
                    float2 pair_3_6 = make_float2(__uint_as_float(v4_1_7[3] << 16), __uint_as_float(v4_1_7[3] & 4294901760u));
                    float2 g_3_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_6) : "l"(*(const unsigned long long*)&pair_3_6), "l"(*(const unsigned long long*)&mm_6));
                    uint32_t _fp4_e2m1_pack_pair_50;
                    uint16_t _e2m1_pair_h_51;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_51) : "f"(g_3_6.x), "f"(g_3_6.y));
                    _fp4_e2m1_pack_pair_50 = (uint32_t)_e2m1_pair_h_51;
                    unsigned int word_6 = fp4_pair_8 | fp4_pair_1_6 << 8 | fp4_pair_2_6 << 16 | _fp4_e2m1_pack_pair_50 << 24;
                    unsigned int carry_6 = (word_6 & 2004318071) + 2004318071;
                    w_1[3] = word_6 & (carry_6 | 2004318071);
                    bool fodd_1_1 = (f & 1) != 0;
                    bool fhi_1_1 = (f & 2) != 0;
                    unsigned int t0_1_1 = w_1[0];
                    unsigned int t1_1_1 = w_1[1];
                    unsigned int t2_1_1 = w_1[2];
                    unsigned int t3_1_1 = w_1[3];
                    if (fodd_1_1) {
                        t0_1_1 = w_1[1];
                        t1_1_1 = w_1[0];
                        t2_1_1 = w_1[3];
                        t3_1_1 = w_1[2];
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
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_1)), "h"((uint16_t)(e4m3x2_f32_0 & 32639)) : "memory");
                    int pp2_1_2 = lane_2 + 64;
                    int base2_1_2 = xb_off + 64 * pp2_1_2;
                    float bm0_1_2 = __uint_as_float(bmp[2] << 16);
                    float bm1_1_2 = __uint_as_float(bmp[2] & 4294901760u);
                    float _fdiv_52 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_2, inv), 6.0f);
                    float _fdiv_53 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_2, inv), 6.0f);
                    uint16_t e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_1) : "f"(_fdiv_52), "f"(_fdiv_53));
                    uint32_t e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_1) : "h"((uint16_t)(e4m3x2_f32_1)));
                    float cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_2) : "h"((uint16_t)(e4m3x2_to_f16x2_1)));
                    float cvt_f32_f16_1_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_1) : "h"((uint16_t)(e4m3x2_to_f16x2_1 >> 16)));
                    float fdiv_rn_2 = __fdiv_rn(inv, cvt_f32_f16_2);
                    float fdiv_rn_1_1 = __fdiv_rn(inv, cvt_f32_f16_1_1);
                    float m0_1 = 0.0f;
                    float m1_1 = 0.0f;
                    if (cvt_f32_f16_2 > 0.0f) {
                        m0_1 = fdiv_rn_2;
                    }
                    if (cvt_f32_f16_1_1 > 0.0f) {
                        m1_1 = fdiv_rn_1_1;
                    }
                    unsigned int w_2[4];
                    int cf_1_8 = f ^ 0;
                    unsigned int v4_1_8[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_8[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_8)));
                    float m_1_8 = m0_1;
                    if ((cf_1_8 & 2) != 0) {
                        m_1_8 = m1_1;
                    }
                    float2 mm_7 = make_float2(m_1_8, m_1_8);
                    float2 pair_10 = make_float2(__uint_as_float(v4_1_8[0] << 16), __uint_as_float(v4_1_8[0] & 4294901760u));
                    float2 g_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_10) : "l"(*(const unsigned long long*)&pair_10), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_9;
                    uint16_t _e2m1_pair_h_54;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_54) : "f"(g_10.x), "f"(g_10.y));
                    fp4_pair_9 = (uint32_t)_e2m1_pair_h_54;
                    float2 pair_1_7 = make_float2(__uint_as_float(v4_1_8[1] << 16), __uint_as_float(v4_1_8[1] & 4294901760u));
                    float2 g_1_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_7) : "l"(*(const unsigned long long*)&pair_1_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_1_7;
                    uint16_t _e2m1_pair_h_55;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_55) : "f"(g_1_7.x), "f"(g_1_7.y));
                    fp4_pair_1_7 = (uint32_t)_e2m1_pair_h_55;
                    float2 pair_2_7 = make_float2(__uint_as_float(v4_1_8[2] << 16), __uint_as_float(v4_1_8[2] & 4294901760u));
                    float2 g_2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_7) : "l"(*(const unsigned long long*)&pair_2_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t fp4_pair_2_7;
                    uint16_t _e2m1_pair_h_56;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_56) : "f"(g_2_7.x), "f"(g_2_7.y));
                    fp4_pair_2_7 = (uint32_t)_e2m1_pair_h_56;
                    float2 pair_3_7 = make_float2(__uint_as_float(v4_1_8[3] << 16), __uint_as_float(v4_1_8[3] & 4294901760u));
                    float2 g_3_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_7) : "l"(*(const unsigned long long*)&pair_3_7), "l"(*(const unsigned long long*)&mm_7));
                    uint32_t _fp4_e2m1_pack_pair_57;
                    uint16_t _e2m1_pair_h_58;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_58) : "f"(g_3_7.x), "f"(g_3_7.y));
                    _fp4_e2m1_pack_pair_57 = (uint32_t)_e2m1_pair_h_58;
                    unsigned int word_7 = fp4_pair_9 | fp4_pair_1_7 << 8 | fp4_pair_2_7 << 16 | _fp4_e2m1_pack_pair_57 << 24;
                    unsigned int carry_7 = (word_7 & 2004318071) + 2004318071;
                    w_2[0] = word_7 & (carry_7 | 2004318071);
                    int cf_1_9 = f ^ 1;
                    unsigned int v4_1_9[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_9[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_9)));
                    float m_1_9 = m0_1;
                    if ((cf_1_9 & 2) != 0) {
                        m_1_9 = m1_1;
                    }
                    float2 mm_8 = make_float2(m_1_9, m_1_9);
                    float2 pair_11 = make_float2(__uint_as_float(v4_1_9[0] << 16), __uint_as_float(v4_1_9[0] & 4294901760u));
                    float2 g_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_11) : "l"(*(const unsigned long long*)&pair_11), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_10;
                    uint16_t _e2m1_pair_h_59;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_59) : "f"(g_11.x), "f"(g_11.y));
                    fp4_pair_10 = (uint32_t)_e2m1_pair_h_59;
                    float2 pair_1_8 = make_float2(__uint_as_float(v4_1_9[1] << 16), __uint_as_float(v4_1_9[1] & 4294901760u));
                    float2 g_1_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_8) : "l"(*(const unsigned long long*)&pair_1_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_1_8;
                    uint16_t _e2m1_pair_h_60;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_60) : "f"(g_1_8.x), "f"(g_1_8.y));
                    fp4_pair_1_8 = (uint32_t)_e2m1_pair_h_60;
                    float2 pair_2_8 = make_float2(__uint_as_float(v4_1_9[2] << 16), __uint_as_float(v4_1_9[2] & 4294901760u));
                    float2 g_2_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_8) : "l"(*(const unsigned long long*)&pair_2_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t fp4_pair_2_8;
                    uint16_t _e2m1_pair_h_61;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_61) : "f"(g_2_8.x), "f"(g_2_8.y));
                    fp4_pair_2_8 = (uint32_t)_e2m1_pair_h_61;
                    float2 pair_3_8 = make_float2(__uint_as_float(v4_1_9[3] << 16), __uint_as_float(v4_1_9[3] & 4294901760u));
                    float2 g_3_8;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_8) : "l"(*(const unsigned long long*)&pair_3_8), "l"(*(const unsigned long long*)&mm_8));
                    uint32_t _fp4_e2m1_pack_pair_62;
                    uint16_t _e2m1_pair_h_63;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_63) : "f"(g_3_8.x), "f"(g_3_8.y));
                    _fp4_e2m1_pack_pair_62 = (uint32_t)_e2m1_pair_h_63;
                    unsigned int word_8 = fp4_pair_10 | fp4_pair_1_8 << 8 | fp4_pair_2_8 << 16 | _fp4_e2m1_pack_pair_62 << 24;
                    unsigned int carry_8 = (word_8 & 2004318071) + 2004318071;
                    w_2[1] = word_8 & (carry_8 | 2004318071);
                    int cf_1_10 = f ^ 2;
                    unsigned int v4_1_10[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_10[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_10)));
                    float m_1_10 = m0_1;
                    if ((cf_1_10 & 2) != 0) {
                        m_1_10 = m1_1;
                    }
                    float2 mm_9 = make_float2(m_1_10, m_1_10);
                    float2 pair_12 = make_float2(__uint_as_float(v4_1_10[0] << 16), __uint_as_float(v4_1_10[0] & 4294901760u));
                    float2 g_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_12) : "l"(*(const unsigned long long*)&pair_12), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_11;
                    uint16_t _e2m1_pair_h_64;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_64) : "f"(g_12.x), "f"(g_12.y));
                    fp4_pair_11 = (uint32_t)_e2m1_pair_h_64;
                    float2 pair_1_9 = make_float2(__uint_as_float(v4_1_10[1] << 16), __uint_as_float(v4_1_10[1] & 4294901760u));
                    float2 g_1_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_9) : "l"(*(const unsigned long long*)&pair_1_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_1_9;
                    uint16_t _e2m1_pair_h_65;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_65) : "f"(g_1_9.x), "f"(g_1_9.y));
                    fp4_pair_1_9 = (uint32_t)_e2m1_pair_h_65;
                    float2 pair_2_9 = make_float2(__uint_as_float(v4_1_10[2] << 16), __uint_as_float(v4_1_10[2] & 4294901760u));
                    float2 g_2_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_9) : "l"(*(const unsigned long long*)&pair_2_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t fp4_pair_2_9;
                    uint16_t _e2m1_pair_h_66;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_66) : "f"(g_2_9.x), "f"(g_2_9.y));
                    fp4_pair_2_9 = (uint32_t)_e2m1_pair_h_66;
                    float2 pair_3_9 = make_float2(__uint_as_float(v4_1_10[3] << 16), __uint_as_float(v4_1_10[3] & 4294901760u));
                    float2 g_3_9;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_9) : "l"(*(const unsigned long long*)&pair_3_9), "l"(*(const unsigned long long*)&mm_9));
                    uint32_t _fp4_e2m1_pack_pair_67;
                    uint16_t _e2m1_pair_h_68;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_68) : "f"(g_3_9.x), "f"(g_3_9.y));
                    _fp4_e2m1_pack_pair_67 = (uint32_t)_e2m1_pair_h_68;
                    unsigned int word_9 = fp4_pair_11 | fp4_pair_1_9 << 8 | fp4_pair_2_9 << 16 | _fp4_e2m1_pack_pair_67 << 24;
                    unsigned int carry_9 = (word_9 & 2004318071) + 2004318071;
                    w_2[2] = word_9 & (carry_9 | 2004318071);
                    int cf_1_11 = f ^ 3;
                    unsigned int v4_1_11[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_11[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_2 + 16 * cf_1_11)));
                    float m_1_11 = m0_1;
                    if ((cf_1_11 & 2) != 0) {
                        m_1_11 = m1_1;
                    }
                    float2 mm_10 = make_float2(m_1_11, m_1_11);
                    float2 pair_13 = make_float2(__uint_as_float(v4_1_11[0] << 16), __uint_as_float(v4_1_11[0] & 4294901760u));
                    float2 g_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_13) : "l"(*(const unsigned long long*)&pair_13), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_12;
                    uint16_t _e2m1_pair_h_69;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_69) : "f"(g_13.x), "f"(g_13.y));
                    fp4_pair_12 = (uint32_t)_e2m1_pair_h_69;
                    float2 pair_1_10 = make_float2(__uint_as_float(v4_1_11[1] << 16), __uint_as_float(v4_1_11[1] & 4294901760u));
                    float2 g_1_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_10) : "l"(*(const unsigned long long*)&pair_1_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_1_10;
                    uint16_t _e2m1_pair_h_70;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_70) : "f"(g_1_10.x), "f"(g_1_10.y));
                    fp4_pair_1_10 = (uint32_t)_e2m1_pair_h_70;
                    float2 pair_2_10 = make_float2(__uint_as_float(v4_1_11[2] << 16), __uint_as_float(v4_1_11[2] & 4294901760u));
                    float2 g_2_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_10) : "l"(*(const unsigned long long*)&pair_2_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t fp4_pair_2_10;
                    uint16_t _e2m1_pair_h_71;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_71) : "f"(g_2_10.x), "f"(g_2_10.y));
                    fp4_pair_2_10 = (uint32_t)_e2m1_pair_h_71;
                    float2 pair_3_10 = make_float2(__uint_as_float(v4_1_11[3] << 16), __uint_as_float(v4_1_11[3] & 4294901760u));
                    float2 g_3_10;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_10) : "l"(*(const unsigned long long*)&pair_3_10), "l"(*(const unsigned long long*)&mm_10));
                    uint32_t _fp4_e2m1_pack_pair_72;
                    uint16_t _e2m1_pair_h_73;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_73) : "f"(g_3_10.x), "f"(g_3_10.y));
                    _fp4_e2m1_pack_pair_72 = (uint32_t)_e2m1_pair_h_73;
                    unsigned int word_10 = fp4_pair_12 | fp4_pair_1_10 << 8 | fp4_pair_2_10 << 16 | _fp4_e2m1_pack_pair_72 << 24;
                    unsigned int carry_10 = (word_10 & 2004318071) + 2004318071;
                    w_2[3] = word_10 & (carry_10 | 2004318071);
                    bool fodd_1_2 = (f & 1) != 0;
                    bool fhi_1_2 = (f & 2) != 0;
                    unsigned int t0_1_2 = w_2[0];
                    unsigned int t1_1_2 = w_2[1];
                    unsigned int t2_1_2 = w_2[2];
                    unsigned int t3_1_2 = w_2[3];
                    if (fodd_1_2) {
                        t0_1_2 = w_2[1];
                        t1_1_2 = w_2[0];
                        t2_1_2 = w_2[3];
                        t3_1_2 = w_2[2];
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
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_2)), "h"((uint16_t)(e4m3x2_f32_1 & 32639)) : "memory");
                    int pp2_1_3 = lane_2 + 96;
                    int base2_1_3 = xb_off + 64 * pp2_1_3;
                    float bm0_1_3 = __uint_as_float(bmp[3] << 16);
                    float bm1_1_3 = __uint_as_float(bmp[3] & 4294901760u);
                    float _fdiv_74 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_3, inv), 6.0f);
                    float _fdiv_75 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_3, inv), 6.0f);
                    uint16_t e4m3x2_f32_2;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_2) : "f"(_fdiv_74), "f"(_fdiv_75));
                    uint32_t e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_2) : "h"((uint16_t)(e4m3x2_f32_2)));
                    float cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_3) : "h"((uint16_t)(e4m3x2_to_f16x2_2)));
                    float cvt_f32_f16_1_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_2) : "h"((uint16_t)(e4m3x2_to_f16x2_2 >> 16)));
                    float fdiv_rn_3 = __fdiv_rn(inv, cvt_f32_f16_3);
                    float fdiv_rn_1_2 = __fdiv_rn(inv, cvt_f32_f16_1_2);
                    float m0_2 = 0.0f;
                    float m1_2 = 0.0f;
                    if (cvt_f32_f16_3 > 0.0f) {
                        m0_2 = fdiv_rn_3;
                    }
                    if (cvt_f32_f16_1_2 > 0.0f) {
                        m1_2 = fdiv_rn_1_2;
                    }
                    unsigned int w_3[4];
                    int cf_1_12 = f ^ 0;
                    unsigned int v4_1_12[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_12[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_12)));
                    float m_1_12 = m0_2;
                    if ((cf_1_12 & 2) != 0) {
                        m_1_12 = m1_2;
                    }
                    float2 mm_11 = make_float2(m_1_12, m_1_12);
                    float2 pair_14 = make_float2(__uint_as_float(v4_1_12[0] << 16), __uint_as_float(v4_1_12[0] & 4294901760u));
                    float2 g_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_14) : "l"(*(const unsigned long long*)&pair_14), "l"(*(const unsigned long long*)&mm_11));
                    uint32_t fp4_pair_13;
                    uint16_t _e2m1_pair_h_76;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_76) : "f"(g_14.x), "f"(g_14.y));
                    fp4_pair_13 = (uint32_t)_e2m1_pair_h_76;
                    float2 pair_1_11 = make_float2(__uint_as_float(v4_1_12[1] << 16), __uint_as_float(v4_1_12[1] & 4294901760u));
                    float2 g_1_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_11) : "l"(*(const unsigned long long*)&pair_1_11), "l"(*(const unsigned long long*)&mm_11));
                    uint32_t fp4_pair_1_11;
                    uint16_t _e2m1_pair_h_77;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_77) : "f"(g_1_11.x), "f"(g_1_11.y));
                    fp4_pair_1_11 = (uint32_t)_e2m1_pair_h_77;
                    float2 pair_2_11 = make_float2(__uint_as_float(v4_1_12[2] << 16), __uint_as_float(v4_1_12[2] & 4294901760u));
                    float2 g_2_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_11) : "l"(*(const unsigned long long*)&pair_2_11), "l"(*(const unsigned long long*)&mm_11));
                    uint32_t fp4_pair_2_11;
                    uint16_t _e2m1_pair_h_78;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_78) : "f"(g_2_11.x), "f"(g_2_11.y));
                    fp4_pair_2_11 = (uint32_t)_e2m1_pair_h_78;
                    float2 pair_3_11 = make_float2(__uint_as_float(v4_1_12[3] << 16), __uint_as_float(v4_1_12[3] & 4294901760u));
                    float2 g_3_11;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_11) : "l"(*(const unsigned long long*)&pair_3_11), "l"(*(const unsigned long long*)&mm_11));
                    uint32_t _fp4_e2m1_pack_pair_79;
                    uint16_t _e2m1_pair_h_80;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_80) : "f"(g_3_11.x), "f"(g_3_11.y));
                    _fp4_e2m1_pack_pair_79 = (uint32_t)_e2m1_pair_h_80;
                    unsigned int word_11 = fp4_pair_13 | fp4_pair_1_11 << 8 | fp4_pair_2_11 << 16 | _fp4_e2m1_pack_pair_79 << 24;
                    unsigned int carry_11 = (word_11 & 2004318071) + 2004318071;
                    w_3[0] = word_11 & (carry_11 | 2004318071);
                    int cf_1_13 = f ^ 1;
                    unsigned int v4_1_13[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_13[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_13)));
                    float m_1_13 = m0_2;
                    if ((cf_1_13 & 2) != 0) {
                        m_1_13 = m1_2;
                    }
                    float2 mm_12 = make_float2(m_1_13, m_1_13);
                    float2 pair_15 = make_float2(__uint_as_float(v4_1_13[0] << 16), __uint_as_float(v4_1_13[0] & 4294901760u));
                    float2 g_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_15) : "l"(*(const unsigned long long*)&pair_15), "l"(*(const unsigned long long*)&mm_12));
                    uint32_t fp4_pair_14;
                    uint16_t _e2m1_pair_h_81;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_81) : "f"(g_15.x), "f"(g_15.y));
                    fp4_pair_14 = (uint32_t)_e2m1_pair_h_81;
                    float2 pair_1_12 = make_float2(__uint_as_float(v4_1_13[1] << 16), __uint_as_float(v4_1_13[1] & 4294901760u));
                    float2 g_1_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_12) : "l"(*(const unsigned long long*)&pair_1_12), "l"(*(const unsigned long long*)&mm_12));
                    uint32_t fp4_pair_1_12;
                    uint16_t _e2m1_pair_h_82;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_82) : "f"(g_1_12.x), "f"(g_1_12.y));
                    fp4_pair_1_12 = (uint32_t)_e2m1_pair_h_82;
                    float2 pair_2_12 = make_float2(__uint_as_float(v4_1_13[2] << 16), __uint_as_float(v4_1_13[2] & 4294901760u));
                    float2 g_2_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_12) : "l"(*(const unsigned long long*)&pair_2_12), "l"(*(const unsigned long long*)&mm_12));
                    uint32_t fp4_pair_2_12;
                    uint16_t _e2m1_pair_h_83;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_83) : "f"(g_2_12.x), "f"(g_2_12.y));
                    fp4_pair_2_12 = (uint32_t)_e2m1_pair_h_83;
                    float2 pair_3_12 = make_float2(__uint_as_float(v4_1_13[3] << 16), __uint_as_float(v4_1_13[3] & 4294901760u));
                    float2 g_3_12;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_12) : "l"(*(const unsigned long long*)&pair_3_12), "l"(*(const unsigned long long*)&mm_12));
                    uint32_t _fp4_e2m1_pack_pair_84;
                    uint16_t _e2m1_pair_h_85;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_85) : "f"(g_3_12.x), "f"(g_3_12.y));
                    _fp4_e2m1_pack_pair_84 = (uint32_t)_e2m1_pair_h_85;
                    unsigned int word_12 = fp4_pair_14 | fp4_pair_1_12 << 8 | fp4_pair_2_12 << 16 | _fp4_e2m1_pack_pair_84 << 24;
                    unsigned int carry_12 = (word_12 & 2004318071) + 2004318071;
                    w_3[1] = word_12 & (carry_12 | 2004318071);
                    int cf_1_14 = f ^ 2;
                    unsigned int v4_1_14[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_14[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_14)));
                    float m_1_14 = m0_2;
                    if ((cf_1_14 & 2) != 0) {
                        m_1_14 = m1_2;
                    }
                    float2 mm_13 = make_float2(m_1_14, m_1_14);
                    float2 pair_16 = make_float2(__uint_as_float(v4_1_14[0] << 16), __uint_as_float(v4_1_14[0] & 4294901760u));
                    float2 g_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_16) : "l"(*(const unsigned long long*)&pair_16), "l"(*(const unsigned long long*)&mm_13));
                    uint32_t fp4_pair_15;
                    uint16_t _e2m1_pair_h_86;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_86) : "f"(g_16.x), "f"(g_16.y));
                    fp4_pair_15 = (uint32_t)_e2m1_pair_h_86;
                    float2 pair_1_13 = make_float2(__uint_as_float(v4_1_14[1] << 16), __uint_as_float(v4_1_14[1] & 4294901760u));
                    float2 g_1_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_13) : "l"(*(const unsigned long long*)&pair_1_13), "l"(*(const unsigned long long*)&mm_13));
                    uint32_t fp4_pair_1_13;
                    uint16_t _e2m1_pair_h_87;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_87) : "f"(g_1_13.x), "f"(g_1_13.y));
                    fp4_pair_1_13 = (uint32_t)_e2m1_pair_h_87;
                    float2 pair_2_13 = make_float2(__uint_as_float(v4_1_14[2] << 16), __uint_as_float(v4_1_14[2] & 4294901760u));
                    float2 g_2_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_13) : "l"(*(const unsigned long long*)&pair_2_13), "l"(*(const unsigned long long*)&mm_13));
                    uint32_t fp4_pair_2_13;
                    uint16_t _e2m1_pair_h_88;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_88) : "f"(g_2_13.x), "f"(g_2_13.y));
                    fp4_pair_2_13 = (uint32_t)_e2m1_pair_h_88;
                    float2 pair_3_13 = make_float2(__uint_as_float(v4_1_14[3] << 16), __uint_as_float(v4_1_14[3] & 4294901760u));
                    float2 g_3_13;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_13) : "l"(*(const unsigned long long*)&pair_3_13), "l"(*(const unsigned long long*)&mm_13));
                    uint32_t _fp4_e2m1_pack_pair_89;
                    uint16_t _e2m1_pair_h_90;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_90) : "f"(g_3_13.x), "f"(g_3_13.y));
                    _fp4_e2m1_pack_pair_89 = (uint32_t)_e2m1_pair_h_90;
                    unsigned int word_13 = fp4_pair_15 | fp4_pair_1_13 << 8 | fp4_pair_2_13 << 16 | _fp4_e2m1_pack_pair_89 << 24;
                    unsigned int carry_13 = (word_13 & 2004318071) + 2004318071;
                    w_3[2] = word_13 & (carry_13 | 2004318071);
                    int cf_1_15 = f ^ 3;
                    unsigned int v4_1_15[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_15[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_3 + 16 * cf_1_15)));
                    float m_1_15 = m0_2;
                    if ((cf_1_15 & 2) != 0) {
                        m_1_15 = m1_2;
                    }
                    float2 mm_14 = make_float2(m_1_15, m_1_15);
                    float2 pair_17 = make_float2(__uint_as_float(v4_1_15[0] << 16), __uint_as_float(v4_1_15[0] & 4294901760u));
                    float2 g_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_17) : "l"(*(const unsigned long long*)&pair_17), "l"(*(const unsigned long long*)&mm_14));
                    uint32_t fp4_pair_16;
                    uint16_t _e2m1_pair_h_91;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_91) : "f"(g_17.x), "f"(g_17.y));
                    fp4_pair_16 = (uint32_t)_e2m1_pair_h_91;
                    float2 pair_1_14 = make_float2(__uint_as_float(v4_1_15[1] << 16), __uint_as_float(v4_1_15[1] & 4294901760u));
                    float2 g_1_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_14) : "l"(*(const unsigned long long*)&pair_1_14), "l"(*(const unsigned long long*)&mm_14));
                    uint32_t fp4_pair_1_14;
                    uint16_t _e2m1_pair_h_92;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_92) : "f"(g_1_14.x), "f"(g_1_14.y));
                    fp4_pair_1_14 = (uint32_t)_e2m1_pair_h_92;
                    float2 pair_2_14 = make_float2(__uint_as_float(v4_1_15[2] << 16), __uint_as_float(v4_1_15[2] & 4294901760u));
                    float2 g_2_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_14) : "l"(*(const unsigned long long*)&pair_2_14), "l"(*(const unsigned long long*)&mm_14));
                    uint32_t fp4_pair_2_14;
                    uint16_t _e2m1_pair_h_93;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_93) : "f"(g_2_14.x), "f"(g_2_14.y));
                    fp4_pair_2_14 = (uint32_t)_e2m1_pair_h_93;
                    float2 pair_3_14 = make_float2(__uint_as_float(v4_1_15[3] << 16), __uint_as_float(v4_1_15[3] & 4294901760u));
                    float2 g_3_14;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_14) : "l"(*(const unsigned long long*)&pair_3_14), "l"(*(const unsigned long long*)&mm_14));
                    uint32_t _fp4_e2m1_pack_pair_94;
                    uint16_t _e2m1_pair_h_95;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_95) : "f"(g_3_14.x), "f"(g_3_14.y));
                    _fp4_e2m1_pack_pair_94 = (uint32_t)_e2m1_pair_h_95;
                    unsigned int word_14 = fp4_pair_16 | fp4_pair_1_14 << 8 | fp4_pair_2_14 << 16 | _fp4_e2m1_pack_pair_94 << 24;
                    unsigned int carry_14 = (word_14 & 2004318071) + 2004318071;
                    w_3[3] = word_14 & (carry_14 | 2004318071);
                    bool fodd_1_3 = (f & 1) != 0;
                    bool fhi_1_3 = (f & 2) != 0;
                    unsigned int t0_1_3 = w_3[0];
                    unsigned int t1_1_3 = w_3[1];
                    unsigned int t2_1_3 = w_3[2];
                    unsigned int t3_1_3 = w_3[3];
                    if (fodd_1_3) {
                        t0_1_3 = w_3[1];
                        t1_1_3 = w_3[0];
                        t2_1_3 = w_3[3];
                        t3_1_3 = w_3[2];
                    }
                    unsigned int q0_1_3 = t0_1_3;
                    unsigned int q1_1_3 = t1_1_3;
                    unsigned int q2_1_3 = t2_1_3;
                    unsigned int q3_1_3 = t3_1_3;
                    if (fhi_1_3) {
                        q0_1_3 = t2_1_3;
                        q1_1_3 = t3_1_3;
                        q2_1_3 = t0_1_3;
                        q3_1_3 = t1_1_3;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_3)), "r"(q0_1_3), "r"(q1_1_3), "r"(q2_1_3), "r"(q3_1_3) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_3)), "h"((uint16_t)(e4m3x2_f32_2 & 32639)) : "memory");
                    int pp2_1_4 = lane_2 + 128;
                    int base2_1_4 = xb_off + 64 * pp2_1_4;
                    float bm0_1_4 = __uint_as_float(bmp[4] << 16);
                    float bm1_1_4 = __uint_as_float(bmp[4] & 4294901760u);
                    float _fdiv_96 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_4, inv), 6.0f);
                    float _fdiv_97 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_4, inv), 6.0f);
                    uint16_t e4m3x2_f32_3;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_3) : "f"(_fdiv_96), "f"(_fdiv_97));
                    uint32_t e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_3) : "h"((uint16_t)(e4m3x2_f32_3)));
                    float cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_4) : "h"((uint16_t)(e4m3x2_to_f16x2_3)));
                    float cvt_f32_f16_1_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_3) : "h"((uint16_t)(e4m3x2_to_f16x2_3 >> 16)));
                    float fdiv_rn_4 = __fdiv_rn(inv, cvt_f32_f16_4);
                    float fdiv_rn_1_3 = __fdiv_rn(inv, cvt_f32_f16_1_3);
                    float m0_3 = 0.0f;
                    float m1_3 = 0.0f;
                    if (cvt_f32_f16_4 > 0.0f) {
                        m0_3 = fdiv_rn_4;
                    }
                    if (cvt_f32_f16_1_3 > 0.0f) {
                        m1_3 = fdiv_rn_1_3;
                    }
                    unsigned int w_4[4];
                    int cf_1_16 = f ^ 0;
                    unsigned int v4_1_16[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_16[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_16)));
                    float m_1_16 = m0_3;
                    if ((cf_1_16 & 2) != 0) {
                        m_1_16 = m1_3;
                    }
                    float2 mm_15 = make_float2(m_1_16, m_1_16);
                    float2 pair_18 = make_float2(__uint_as_float(v4_1_16[0] << 16), __uint_as_float(v4_1_16[0] & 4294901760u));
                    float2 g_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_18) : "l"(*(const unsigned long long*)&pair_18), "l"(*(const unsigned long long*)&mm_15));
                    uint32_t fp4_pair_17;
                    uint16_t _e2m1_pair_h_98;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_98) : "f"(g_18.x), "f"(g_18.y));
                    fp4_pair_17 = (uint32_t)_e2m1_pair_h_98;
                    float2 pair_1_15 = make_float2(__uint_as_float(v4_1_16[1] << 16), __uint_as_float(v4_1_16[1] & 4294901760u));
                    float2 g_1_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_15) : "l"(*(const unsigned long long*)&pair_1_15), "l"(*(const unsigned long long*)&mm_15));
                    uint32_t fp4_pair_1_15;
                    uint16_t _e2m1_pair_h_99;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_99) : "f"(g_1_15.x), "f"(g_1_15.y));
                    fp4_pair_1_15 = (uint32_t)_e2m1_pair_h_99;
                    float2 pair_2_15 = make_float2(__uint_as_float(v4_1_16[2] << 16), __uint_as_float(v4_1_16[2] & 4294901760u));
                    float2 g_2_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_15) : "l"(*(const unsigned long long*)&pair_2_15), "l"(*(const unsigned long long*)&mm_15));
                    uint32_t fp4_pair_2_15;
                    uint16_t _e2m1_pair_h_100;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_100) : "f"(g_2_15.x), "f"(g_2_15.y));
                    fp4_pair_2_15 = (uint32_t)_e2m1_pair_h_100;
                    float2 pair_3_15 = make_float2(__uint_as_float(v4_1_16[3] << 16), __uint_as_float(v4_1_16[3] & 4294901760u));
                    float2 g_3_15;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_15) : "l"(*(const unsigned long long*)&pair_3_15), "l"(*(const unsigned long long*)&mm_15));
                    uint32_t _fp4_e2m1_pack_pair_101;
                    uint16_t _e2m1_pair_h_102;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_102) : "f"(g_3_15.x), "f"(g_3_15.y));
                    _fp4_e2m1_pack_pair_101 = (uint32_t)_e2m1_pair_h_102;
                    unsigned int word_15 = fp4_pair_17 | fp4_pair_1_15 << 8 | fp4_pair_2_15 << 16 | _fp4_e2m1_pack_pair_101 << 24;
                    unsigned int carry_15 = (word_15 & 2004318071) + 2004318071;
                    w_4[0] = word_15 & (carry_15 | 2004318071);
                    int cf_1_17 = f ^ 1;
                    unsigned int v4_1_17[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_17[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_17)));
                    float m_1_17 = m0_3;
                    if ((cf_1_17 & 2) != 0) {
                        m_1_17 = m1_3;
                    }
                    float2 mm_16 = make_float2(m_1_17, m_1_17);
                    float2 pair_19 = make_float2(__uint_as_float(v4_1_17[0] << 16), __uint_as_float(v4_1_17[0] & 4294901760u));
                    float2 g_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_19) : "l"(*(const unsigned long long*)&pair_19), "l"(*(const unsigned long long*)&mm_16));
                    uint32_t fp4_pair_18;
                    uint16_t _e2m1_pair_h_103;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_103) : "f"(g_19.x), "f"(g_19.y));
                    fp4_pair_18 = (uint32_t)_e2m1_pair_h_103;
                    float2 pair_1_16 = make_float2(__uint_as_float(v4_1_17[1] << 16), __uint_as_float(v4_1_17[1] & 4294901760u));
                    float2 g_1_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_16) : "l"(*(const unsigned long long*)&pair_1_16), "l"(*(const unsigned long long*)&mm_16));
                    uint32_t fp4_pair_1_16;
                    uint16_t _e2m1_pair_h_104;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_104) : "f"(g_1_16.x), "f"(g_1_16.y));
                    fp4_pair_1_16 = (uint32_t)_e2m1_pair_h_104;
                    float2 pair_2_16 = make_float2(__uint_as_float(v4_1_17[2] << 16), __uint_as_float(v4_1_17[2] & 4294901760u));
                    float2 g_2_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_16) : "l"(*(const unsigned long long*)&pair_2_16), "l"(*(const unsigned long long*)&mm_16));
                    uint32_t fp4_pair_2_16;
                    uint16_t _e2m1_pair_h_105;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_105) : "f"(g_2_16.x), "f"(g_2_16.y));
                    fp4_pair_2_16 = (uint32_t)_e2m1_pair_h_105;
                    float2 pair_3_16 = make_float2(__uint_as_float(v4_1_17[3] << 16), __uint_as_float(v4_1_17[3] & 4294901760u));
                    float2 g_3_16;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_16) : "l"(*(const unsigned long long*)&pair_3_16), "l"(*(const unsigned long long*)&mm_16));
                    uint32_t _fp4_e2m1_pack_pair_106;
                    uint16_t _e2m1_pair_h_107;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_107) : "f"(g_3_16.x), "f"(g_3_16.y));
                    _fp4_e2m1_pack_pair_106 = (uint32_t)_e2m1_pair_h_107;
                    unsigned int word_16 = fp4_pair_18 | fp4_pair_1_16 << 8 | fp4_pair_2_16 << 16 | _fp4_e2m1_pack_pair_106 << 24;
                    unsigned int carry_16 = (word_16 & 2004318071) + 2004318071;
                    w_4[1] = word_16 & (carry_16 | 2004318071);
                    int cf_1_18 = f ^ 2;
                    unsigned int v4_1_18[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_18[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_18)));
                    float m_1_18 = m0_3;
                    if ((cf_1_18 & 2) != 0) {
                        m_1_18 = m1_3;
                    }
                    float2 mm_17 = make_float2(m_1_18, m_1_18);
                    float2 pair_20 = make_float2(__uint_as_float(v4_1_18[0] << 16), __uint_as_float(v4_1_18[0] & 4294901760u));
                    float2 g_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_20) : "l"(*(const unsigned long long*)&pair_20), "l"(*(const unsigned long long*)&mm_17));
                    uint32_t fp4_pair_19;
                    uint16_t _e2m1_pair_h_108;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_108) : "f"(g_20.x), "f"(g_20.y));
                    fp4_pair_19 = (uint32_t)_e2m1_pair_h_108;
                    float2 pair_1_17 = make_float2(__uint_as_float(v4_1_18[1] << 16), __uint_as_float(v4_1_18[1] & 4294901760u));
                    float2 g_1_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_17) : "l"(*(const unsigned long long*)&pair_1_17), "l"(*(const unsigned long long*)&mm_17));
                    uint32_t fp4_pair_1_17;
                    uint16_t _e2m1_pair_h_109;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_109) : "f"(g_1_17.x), "f"(g_1_17.y));
                    fp4_pair_1_17 = (uint32_t)_e2m1_pair_h_109;
                    float2 pair_2_17 = make_float2(__uint_as_float(v4_1_18[2] << 16), __uint_as_float(v4_1_18[2] & 4294901760u));
                    float2 g_2_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_17) : "l"(*(const unsigned long long*)&pair_2_17), "l"(*(const unsigned long long*)&mm_17));
                    uint32_t fp4_pair_2_17;
                    uint16_t _e2m1_pair_h_110;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_110) : "f"(g_2_17.x), "f"(g_2_17.y));
                    fp4_pair_2_17 = (uint32_t)_e2m1_pair_h_110;
                    float2 pair_3_17 = make_float2(__uint_as_float(v4_1_18[3] << 16), __uint_as_float(v4_1_18[3] & 4294901760u));
                    float2 g_3_17;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_17) : "l"(*(const unsigned long long*)&pair_3_17), "l"(*(const unsigned long long*)&mm_17));
                    uint32_t _fp4_e2m1_pack_pair_111;
                    uint16_t _e2m1_pair_h_112;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_112) : "f"(g_3_17.x), "f"(g_3_17.y));
                    _fp4_e2m1_pack_pair_111 = (uint32_t)_e2m1_pair_h_112;
                    unsigned int word_17 = fp4_pair_19 | fp4_pair_1_17 << 8 | fp4_pair_2_17 << 16 | _fp4_e2m1_pack_pair_111 << 24;
                    unsigned int carry_17 = (word_17 & 2004318071) + 2004318071;
                    w_4[2] = word_17 & (carry_17 | 2004318071);
                    int cf_1_19 = f ^ 3;
                    unsigned int v4_1_19[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_19[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_4 + 16 * cf_1_19)));
                    float m_1_19 = m0_3;
                    if ((cf_1_19 & 2) != 0) {
                        m_1_19 = m1_3;
                    }
                    float2 mm_18 = make_float2(m_1_19, m_1_19);
                    float2 pair_21 = make_float2(__uint_as_float(v4_1_19[0] << 16), __uint_as_float(v4_1_19[0] & 4294901760u));
                    float2 g_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_21) : "l"(*(const unsigned long long*)&pair_21), "l"(*(const unsigned long long*)&mm_18));
                    uint32_t fp4_pair_20;
                    uint16_t _e2m1_pair_h_113;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_113) : "f"(g_21.x), "f"(g_21.y));
                    fp4_pair_20 = (uint32_t)_e2m1_pair_h_113;
                    float2 pair_1_18 = make_float2(__uint_as_float(v4_1_19[1] << 16), __uint_as_float(v4_1_19[1] & 4294901760u));
                    float2 g_1_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_18) : "l"(*(const unsigned long long*)&pair_1_18), "l"(*(const unsigned long long*)&mm_18));
                    uint32_t fp4_pair_1_18;
                    uint16_t _e2m1_pair_h_114;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_114) : "f"(g_1_18.x), "f"(g_1_18.y));
                    fp4_pair_1_18 = (uint32_t)_e2m1_pair_h_114;
                    float2 pair_2_18 = make_float2(__uint_as_float(v4_1_19[2] << 16), __uint_as_float(v4_1_19[2] & 4294901760u));
                    float2 g_2_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_18) : "l"(*(const unsigned long long*)&pair_2_18), "l"(*(const unsigned long long*)&mm_18));
                    uint32_t fp4_pair_2_18;
                    uint16_t _e2m1_pair_h_115;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_115) : "f"(g_2_18.x), "f"(g_2_18.y));
                    fp4_pair_2_18 = (uint32_t)_e2m1_pair_h_115;
                    float2 pair_3_18 = make_float2(__uint_as_float(v4_1_19[3] << 16), __uint_as_float(v4_1_19[3] & 4294901760u));
                    float2 g_3_18;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_18) : "l"(*(const unsigned long long*)&pair_3_18), "l"(*(const unsigned long long*)&mm_18));
                    uint32_t _fp4_e2m1_pack_pair_116;
                    uint16_t _e2m1_pair_h_117;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_117) : "f"(g_3_18.x), "f"(g_3_18.y));
                    _fp4_e2m1_pack_pair_116 = (uint32_t)_e2m1_pair_h_117;
                    unsigned int word_18 = fp4_pair_20 | fp4_pair_1_18 << 8 | fp4_pair_2_18 << 16 | _fp4_e2m1_pack_pair_116 << 24;
                    unsigned int carry_18 = (word_18 & 2004318071) + 2004318071;
                    w_4[3] = word_18 & (carry_18 | 2004318071);
                    bool fodd_1_4 = (f & 1) != 0;
                    bool fhi_1_4 = (f & 2) != 0;
                    unsigned int t0_1_4 = w_4[0];
                    unsigned int t1_1_4 = w_4[1];
                    unsigned int t2_1_4 = w_4[2];
                    unsigned int t3_1_4 = w_4[3];
                    if (fodd_1_4) {
                        t0_1_4 = w_4[1];
                        t1_1_4 = w_4[0];
                        t2_1_4 = w_4[3];
                        t3_1_4 = w_4[2];
                    }
                    unsigned int q0_1_4 = t0_1_4;
                    unsigned int q1_1_4 = t1_1_4;
                    unsigned int q2_1_4 = t2_1_4;
                    unsigned int q3_1_4 = t3_1_4;
                    if (fhi_1_4) {
                        q0_1_4 = t2_1_4;
                        q1_1_4 = t3_1_4;
                        q2_1_4 = t0_1_4;
                        q3_1_4 = t1_1_4;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_4)), "r"(q0_1_4), "r"(q1_1_4), "r"(q2_1_4), "r"(q3_1_4) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_4)), "h"((uint16_t)(e4m3x2_f32_3 & 32639)) : "memory");
                    int pp2_1_5 = lane_2 + 160;
                    int base2_1_5 = xb_off + 64 * pp2_1_5;
                    float bm0_1_5 = __uint_as_float(bmp[5] << 16);
                    float bm1_1_5 = __uint_as_float(bmp[5] & 4294901760u);
                    float _fdiv_118 = __fdiv_rn(mul_f32_rn_noftz(bm1_1_5, inv), 6.0f);
                    float _fdiv_119 = __fdiv_rn(mul_f32_rn_noftz(bm0_1_5, inv), 6.0f);
                    uint16_t e4m3x2_f32_4;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_f32_4) : "f"(_fdiv_118), "f"(_fdiv_119));
                    uint32_t e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_4) : "h"((uint16_t)(e4m3x2_f32_4)));
                    float cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_5) : "h"((uint16_t)(e4m3x2_to_f16x2_4)));
                    float cvt_f32_f16_1_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_4) : "h"((uint16_t)(e4m3x2_to_f16x2_4 >> 16)));
                    float fdiv_rn_5 = __fdiv_rn(inv, cvt_f32_f16_5);
                    float fdiv_rn_1_4 = __fdiv_rn(inv, cvt_f32_f16_1_4);
                    float m0_4 = 0.0f;
                    float m1_4 = 0.0f;
                    if (cvt_f32_f16_5 > 0.0f) {
                        m0_4 = fdiv_rn_5;
                    }
                    if (cvt_f32_f16_1_4 > 0.0f) {
                        m1_4 = fdiv_rn_1_4;
                    }
                    unsigned int w_5[4];
                    int cf_1_20 = f ^ 0;
                    unsigned int v4_1_20[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_20[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_20)));
                    float m_1_20 = m0_4;
                    if ((cf_1_20 & 2) != 0) {
                        m_1_20 = m1_4;
                    }
                    float2 mm_19 = make_float2(m_1_20, m_1_20);
                    float2 pair_22 = make_float2(__uint_as_float(v4_1_20[0] << 16), __uint_as_float(v4_1_20[0] & 4294901760u));
                    float2 g_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_22) : "l"(*(const unsigned long long*)&pair_22), "l"(*(const unsigned long long*)&mm_19));
                    uint32_t fp4_pair_21;
                    uint16_t _e2m1_pair_h_120;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_120) : "f"(g_22.x), "f"(g_22.y));
                    fp4_pair_21 = (uint32_t)_e2m1_pair_h_120;
                    float2 pair_1_19 = make_float2(__uint_as_float(v4_1_20[1] << 16), __uint_as_float(v4_1_20[1] & 4294901760u));
                    float2 g_1_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_19) : "l"(*(const unsigned long long*)&pair_1_19), "l"(*(const unsigned long long*)&mm_19));
                    uint32_t fp4_pair_1_19;
                    uint16_t _e2m1_pair_h_121;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_121) : "f"(g_1_19.x), "f"(g_1_19.y));
                    fp4_pair_1_19 = (uint32_t)_e2m1_pair_h_121;
                    float2 pair_2_19 = make_float2(__uint_as_float(v4_1_20[2] << 16), __uint_as_float(v4_1_20[2] & 4294901760u));
                    float2 g_2_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_19) : "l"(*(const unsigned long long*)&pair_2_19), "l"(*(const unsigned long long*)&mm_19));
                    uint32_t fp4_pair_2_19;
                    uint16_t _e2m1_pair_h_122;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_122) : "f"(g_2_19.x), "f"(g_2_19.y));
                    fp4_pair_2_19 = (uint32_t)_e2m1_pair_h_122;
                    float2 pair_3_19 = make_float2(__uint_as_float(v4_1_20[3] << 16), __uint_as_float(v4_1_20[3] & 4294901760u));
                    float2 g_3_19;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_19) : "l"(*(const unsigned long long*)&pair_3_19), "l"(*(const unsigned long long*)&mm_19));
                    uint32_t _fp4_e2m1_pack_pair_123;
                    uint16_t _e2m1_pair_h_124;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_124) : "f"(g_3_19.x), "f"(g_3_19.y));
                    _fp4_e2m1_pack_pair_123 = (uint32_t)_e2m1_pair_h_124;
                    unsigned int word_19 = fp4_pair_21 | fp4_pair_1_19 << 8 | fp4_pair_2_19 << 16 | _fp4_e2m1_pack_pair_123 << 24;
                    unsigned int carry_19 = (word_19 & 2004318071) + 2004318071;
                    w_5[0] = word_19 & (carry_19 | 2004318071);
                    int cf_1_21 = f ^ 1;
                    unsigned int v4_1_21[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_21[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_21)));
                    float m_1_21 = m0_4;
                    if ((cf_1_21 & 2) != 0) {
                        m_1_21 = m1_4;
                    }
                    float2 mm_20 = make_float2(m_1_21, m_1_21);
                    float2 pair_23 = make_float2(__uint_as_float(v4_1_21[0] << 16), __uint_as_float(v4_1_21[0] & 4294901760u));
                    float2 g_23;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_23) : "l"(*(const unsigned long long*)&pair_23), "l"(*(const unsigned long long*)&mm_20));
                    uint32_t fp4_pair_22;
                    uint16_t _e2m1_pair_h_125;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_125) : "f"(g_23.x), "f"(g_23.y));
                    fp4_pair_22 = (uint32_t)_e2m1_pair_h_125;
                    float2 pair_1_20 = make_float2(__uint_as_float(v4_1_21[1] << 16), __uint_as_float(v4_1_21[1] & 4294901760u));
                    float2 g_1_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_20) : "l"(*(const unsigned long long*)&pair_1_20), "l"(*(const unsigned long long*)&mm_20));
                    uint32_t fp4_pair_1_20;
                    uint16_t _e2m1_pair_h_126;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_126) : "f"(g_1_20.x), "f"(g_1_20.y));
                    fp4_pair_1_20 = (uint32_t)_e2m1_pair_h_126;
                    float2 pair_2_20 = make_float2(__uint_as_float(v4_1_21[2] << 16), __uint_as_float(v4_1_21[2] & 4294901760u));
                    float2 g_2_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_20) : "l"(*(const unsigned long long*)&pair_2_20), "l"(*(const unsigned long long*)&mm_20));
                    uint32_t fp4_pair_2_20;
                    uint16_t _e2m1_pair_h_127;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_127) : "f"(g_2_20.x), "f"(g_2_20.y));
                    fp4_pair_2_20 = (uint32_t)_e2m1_pair_h_127;
                    float2 pair_3_20 = make_float2(__uint_as_float(v4_1_21[3] << 16), __uint_as_float(v4_1_21[3] & 4294901760u));
                    float2 g_3_20;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_20) : "l"(*(const unsigned long long*)&pair_3_20), "l"(*(const unsigned long long*)&mm_20));
                    uint32_t _fp4_e2m1_pack_pair_128;
                    uint16_t _e2m1_pair_h_129;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_129) : "f"(g_3_20.x), "f"(g_3_20.y));
                    _fp4_e2m1_pack_pair_128 = (uint32_t)_e2m1_pair_h_129;
                    unsigned int word_20 = fp4_pair_22 | fp4_pair_1_20 << 8 | fp4_pair_2_20 << 16 | _fp4_e2m1_pack_pair_128 << 24;
                    unsigned int carry_20 = (word_20 & 2004318071) + 2004318071;
                    w_5[1] = word_20 & (carry_20 | 2004318071);
                    int cf_1_22 = f ^ 2;
                    unsigned int v4_1_22[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_22[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_22)));
                    float m_1_22 = m0_4;
                    if ((cf_1_22 & 2) != 0) {
                        m_1_22 = m1_4;
                    }
                    float2 mm_21 = make_float2(m_1_22, m_1_22);
                    float2 pair_24 = make_float2(__uint_as_float(v4_1_22[0] << 16), __uint_as_float(v4_1_22[0] & 4294901760u));
                    float2 g_24;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_24) : "l"(*(const unsigned long long*)&pair_24), "l"(*(const unsigned long long*)&mm_21));
                    uint32_t fp4_pair_23;
                    uint16_t _e2m1_pair_h_130;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_130) : "f"(g_24.x), "f"(g_24.y));
                    fp4_pair_23 = (uint32_t)_e2m1_pair_h_130;
                    float2 pair_1_21 = make_float2(__uint_as_float(v4_1_22[1] << 16), __uint_as_float(v4_1_22[1] & 4294901760u));
                    float2 g_1_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_21) : "l"(*(const unsigned long long*)&pair_1_21), "l"(*(const unsigned long long*)&mm_21));
                    uint32_t fp4_pair_1_21;
                    uint16_t _e2m1_pair_h_131;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_131) : "f"(g_1_21.x), "f"(g_1_21.y));
                    fp4_pair_1_21 = (uint32_t)_e2m1_pair_h_131;
                    float2 pair_2_21 = make_float2(__uint_as_float(v4_1_22[2] << 16), __uint_as_float(v4_1_22[2] & 4294901760u));
                    float2 g_2_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_21) : "l"(*(const unsigned long long*)&pair_2_21), "l"(*(const unsigned long long*)&mm_21));
                    uint32_t fp4_pair_2_21;
                    uint16_t _e2m1_pair_h_132;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_132) : "f"(g_2_21.x), "f"(g_2_21.y));
                    fp4_pair_2_21 = (uint32_t)_e2m1_pair_h_132;
                    float2 pair_3_21 = make_float2(__uint_as_float(v4_1_22[3] << 16), __uint_as_float(v4_1_22[3] & 4294901760u));
                    float2 g_3_21;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_21) : "l"(*(const unsigned long long*)&pair_3_21), "l"(*(const unsigned long long*)&mm_21));
                    uint32_t _fp4_e2m1_pack_pair_133;
                    uint16_t _e2m1_pair_h_134;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_134) : "f"(g_3_21.x), "f"(g_3_21.y));
                    _fp4_e2m1_pack_pair_133 = (uint32_t)_e2m1_pair_h_134;
                    unsigned int word_21 = fp4_pair_23 | fp4_pair_1_21 << 8 | fp4_pair_2_21 << 16 | _fp4_e2m1_pack_pair_133 << 24;
                    unsigned int carry_21 = (word_21 & 2004318071) + 2004318071;
                    w_5[2] = word_21 & (carry_21 | 2004318071);
                    int cf_1_23 = f ^ 3;
                    unsigned int v4_1_23[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&v4_1_23[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(base2_1_5 + 16 * cf_1_23)));
                    float m_1_23 = m0_4;
                    if ((cf_1_23 & 2) != 0) {
                        m_1_23 = m1_4;
                    }
                    float2 mm_22 = make_float2(m_1_23, m_1_23);
                    float2 pair_25 = make_float2(__uint_as_float(v4_1_23[0] << 16), __uint_as_float(v4_1_23[0] & 4294901760u));
                    float2 g_25;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_25) : "l"(*(const unsigned long long*)&pair_25), "l"(*(const unsigned long long*)&mm_22));
                    uint32_t fp4_pair_24;
                    uint16_t _e2m1_pair_h_135;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_135) : "f"(g_25.x), "f"(g_25.y));
                    fp4_pair_24 = (uint32_t)_e2m1_pair_h_135;
                    float2 pair_1_22 = make_float2(__uint_as_float(v4_1_23[1] << 16), __uint_as_float(v4_1_23[1] & 4294901760u));
                    float2 g_1_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_1_22) : "l"(*(const unsigned long long*)&pair_1_22), "l"(*(const unsigned long long*)&mm_22));
                    uint32_t fp4_pair_1_22;
                    uint16_t _e2m1_pair_h_136;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_136) : "f"(g_1_22.x), "f"(g_1_22.y));
                    fp4_pair_1_22 = (uint32_t)_e2m1_pair_h_136;
                    float2 pair_2_22 = make_float2(__uint_as_float(v4_1_23[2] << 16), __uint_as_float(v4_1_23[2] & 4294901760u));
                    float2 g_2_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_2_22) : "l"(*(const unsigned long long*)&pair_2_22), "l"(*(const unsigned long long*)&mm_22));
                    uint32_t fp4_pair_2_22;
                    uint16_t _e2m1_pair_h_137;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_137) : "f"(g_2_22.x), "f"(g_2_22.y));
                    fp4_pair_2_22 = (uint32_t)_e2m1_pair_h_137;
                    float2 pair_3_22 = make_float2(__uint_as_float(v4_1_23[3] << 16), __uint_as_float(v4_1_23[3] & 4294901760u));
                    float2 g_3_22;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_3_22) : "l"(*(const unsigned long long*)&pair_3_22), "l"(*(const unsigned long long*)&mm_22));
                    uint32_t _fp4_e2m1_pack_pair_138;
                    uint16_t _e2m1_pair_h_139;
                    asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_139) : "f"(g_3_22.x), "f"(g_3_22.y));
                    _fp4_e2m1_pack_pair_138 = (uint32_t)_e2m1_pair_h_139;
                    unsigned int word_22 = fp4_pair_24 | fp4_pair_1_22 << 8 | fp4_pair_2_22 << 16 | _fp4_e2m1_pack_pair_138 << 24;
                    unsigned int carry_22 = (word_22 & 2004318071) + 2004318071;
                    w_5[3] = word_22 & (carry_22 | 2004318071);
                    bool fodd_1_5 = (f & 1) != 0;
                    bool fhi_1_5 = (f & 2) != 0;
                    unsigned int t0_1_5 = w_5[0];
                    unsigned int t1_1_5 = w_5[1];
                    unsigned int t2_1_5 = w_5[2];
                    unsigned int t3_1_5 = w_5[3];
                    if (fodd_1_5) {
                        t0_1_5 = w_5[1];
                        t1_1_5 = w_5[0];
                        t2_1_5 = w_5[3];
                        t3_1_5 = w_5[2];
                    }
                    unsigned int q0_1_5 = t0_1_5;
                    unsigned int q1_1_5 = t1_1_5;
                    unsigned int q2_1_5 = t2_1_5;
                    unsigned int q3_1_5 = t3_1_5;
                    if (fhi_1_5) {
                        q0_1_5 = t2_1_5;
                        q1_1_5 = t3_1_5;
                        q2_1_5 = t0_1_5;
                        q3_1_5 = t1_1_5;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(ob_off + 16 * pp2_1_5)), "r"(q0_1_5), "r"(q1_1_5), "r"(q2_1_5), "r"(q3_1_5) : "memory");
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(wsm_addr + (unsigned int)(ob_off + 3072 + 2 * pp2_1_5)), "h"((uint16_t)(e4m3x2_f32_4 & 32639)) : "memory");
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
                    long long drow = (long long)(r0 + warp_1 + 11 * kd);
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
                    uint32_t word_23;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(word_23) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = word_23;
                    float ssx = 0.0f;
                    float ssy = 0.0f;
                    #pragma unroll 1
                    for (int j0 = 0; j0 < 6; j0 += 2) {
                        int pp0 = lane_2 + 32 * j0;
                        int pp1 = pp0 + 32;
                        uint32_t values[8];
                        uint32_t scales[2];
                        uint16_t _wire_h16_144_0;
                        uint16_t _wire_h16_144_1;
                        uint32_t _wire_vpend_144 = 3u;
                        uint32_t _wire_spend_144 = 3u;
                        do {
                            if (_wire_vpend_144 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0)))) : "memory");
                            }
                            if (_wire_vpend_144 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(values[4]), "=r"(values[5]), "=r"(values[6]), "=r"(values[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1)))) : "memory");
                            }
                            if (_wire_spend_144 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_144_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0)))) : "memory");
                                scales[0] = (uint32_t)_wire_h16_144_0;
                            }
                            if (_wire_spend_144 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_144_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1)))) : "memory");
                                scales[1] = (uint32_t)_wire_h16_144_1;
                            }
                            _wire_vpend_144 = ((uint32_t)((((values[0] & ~((values[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[1] & ~((values[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[2] & ~((values[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[3] & ~((values[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((values[4] & ~((values[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[5] & ~((values[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[6] & ~((values[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((values[7] & ~((values[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_144 = ((uint32_t)(((uint32_t)((scales[0] & 0xFFu) == 0xFFu) | (uint32_t)(((scales[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((scales[1] & 0xFFu) == 0xFFu) | (uint32_t)(((scales[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
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
                        unsigned int a_1[4];
                        unsigned int b_1[4];
                        a_1[0] = ((first) ? own[0] : values[0]);
                        b_1[0] = ((first) ? values[0] : own[0]);
                        a_1[1] = ((first) ? own[1] : values[1]);
                        b_1[1] = ((first) ? values[1] : own[1]);
                        a_1[2] = ((first) ? own[2] : values[2]);
                        b_1[2] = ((first) ? values[2] : own[2]);
                        a_1[3] = ((first) ? own[3] : values[3]);
                        b_1[3] = ((first) ? values[3] : own[3]);
                        unsigned int a_sc = ((first) ? own_sc[0] : scales[0]);
                        unsigned int b_sc = ((first) ? scales[0] : own_sc[0]);
                        float acc[32] = {0};
                        bool fhi = (f & 2) != 0;
                        bool fodd = (f & 1) != 0;
                        bool fhi_0 = (f & 2) != 0;
                        unsigned int s[4];
                        s[0] = a_1[0];
                        s[1] = a_1[1];
                        s[2] = a_1[2];
                        s[3] = a_1[3];
                        if (fodd) {
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
                        bool fodd_0 = (f & 1) != 0;
                        bool fhi_1 = (f & 2) != 0;
                        unsigned int s_0[4];
                        s_0[0] = b_1[0];
                        s_0[1] = b_1[1];
                        s_0[2] = b_1[2];
                        s_0[3] = b_1[3];
                        if (fodd_0) {
                            s_0[0] = b_1[1];
                            s_0[1] = b_1[0];
                            s_0[2] = b_1[3];
                            s_0[3] = b_1[2];
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
                        uint32_t e4m3x2_to_f16x2_5;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5) : "h"((uint16_t)(a_sc)));
                        float cvt_f32_f16_6;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6) : "h"((uint16_t)(e4m3x2_to_f16x2_5)));
                        float cvt_f32_f16_1_5;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5) : "h"((uint16_t)(e4m3x2_to_f16x2_5 >> 16)));
                        float s0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0) : "f"(cvt_f32_f16_6), "f"(rs_a_f));
                        float s1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1) : "f"(cvt_f32_f16_1_5), "f"(rs_a_f));
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
                        float2 a_1_1 = make_float2(acc[2], acc[3]);
                        float2 q_1 = make_float2(tv_0[2], tv_0[3]);
                        float2 r_2 = fma_f32x2_rn_noftz(q_1, scale0, a_1_1);
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
                        uint32_t e4m3x2_to_f16x2_5_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_5_1) : "h"((uint16_t)(b_sc)));
                        float cvt_f32_f16_6_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_6_1) : "h"((uint16_t)(e4m3x2_to_f16x2_5_1)));
                        float cvt_f32_f16_1_5_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_5_1) : "h"((uint16_t)(e4m3x2_to_f16x2_5_1 >> 16)));
                        float s0_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_0) : "f"(cvt_f32_f16_6_1), "f"(rs_b_f));
                        float s1_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_0) : "f"(cvt_f32_f16_1_5_1), "f"(rs_b_f));
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
                        uint32_t e4m3x2_to_f16x2_6;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_6) : "h"((uint16_t)(a_sc_0)));
                        float cvt_f32_f16_7;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_7) : "h"((uint16_t)(e4m3x2_to_f16x2_6)));
                        float cvt_f32_f16_1_6;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_6) : "h"((uint16_t)(e4m3x2_to_f16x2_6 >> 16)));
                        float s0_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_1) : "f"(cvt_f32_f16_7), "f"(rs_a_f));
                        float s1_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_1) : "f"(cvt_f32_f16_1_6), "f"(rs_a_f));
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
                        uint32_t e4m3x2_to_f16x2_7;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_7) : "h"((uint16_t)(b_sc_0)));
                        float cvt_f32_f16_8;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_8) : "h"((uint16_t)(e4m3x2_to_f16x2_7)));
                        float cvt_f32_f16_1_7;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_7) : "h"((uint16_t)(e4m3x2_to_f16x2_7 >> 16)));
                        float s0_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s0_2) : "f"(cvt_f32_f16_8), "f"(rs_b_f));
                        float s1_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(s1_2) : "f"(cvt_f32_f16_1_7), "f"(rs_b_f));
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
                        float2 tz_0;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&r2));
                        float2 tw_0 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[0] << 16), __uint_as_float(tw[0] & 4294901760u)));
                        float2 t_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&tz_0), "l"(*(const unsigned long long*)&tw_0));
                        y[0] = t_1.x;
                        y[1] = t_1.y;
                        float2 _f32x2_mul_a_158 = make_float2(__uint_as_float(tx[1] << 16), __uint_as_float(tx[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&r2));
                        float2 tw_1 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[1] << 16), __uint_as_float(tw[1] & 4294901760u)));
                        float2 t_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_1));
                        y[2] = t_2.x;
                        y[3] = t_2.y;
                        float2 _f32x2_mul_a_159 = make_float2(__uint_as_float(tx[2] << 16), __uint_as_float(tx[2] & 4294901760u));
                        float2 tz_2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_159), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[2] << 16), __uint_as_float(tw[2] & 4294901760u)));
                        float2 t_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&tz_2), "l"(*(const unsigned long long*)&tw_2));
                        y[4] = t_3.x;
                        y[5] = t_3.y;
                        float2 _f32x2_mul_a_160 = make_float2(__uint_as_float(tx[3] << 16), __uint_as_float(tx[3] & 4294901760u));
                        float2 tz_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&r2));
                        float2 tw_3 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[3] << 16), __uint_as_float(tw[3] & 4294901760u)));
                        float2 t_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&tz_3), "l"(*(const unsigned long long*)&tw_3));
                        y[6] = t_4.x;
                        y[7] = t_4.y;
                        float2 _f32x2_mul_a_161 = make_float2(__uint_as_float(tx[4] << 16), __uint_as_float(tx[4] & 4294901760u));
                        float2 tz_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_161), "l"(*(const unsigned long long*)&r2));
                        float2 tw_4 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[4] << 16), __uint_as_float(tw[4] & 4294901760u)));
                        float2 t_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&tz_4), "l"(*(const unsigned long long*)&tw_4));
                        y[8] = t_5.x;
                        y[9] = t_5.y;
                        float2 _f32x2_mul_a_162 = make_float2(__uint_as_float(tx[5] << 16), __uint_as_float(tx[5] & 4294901760u));
                        float2 tz_5;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&r2));
                        float2 tw_5 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[5] << 16), __uint_as_float(tw[5] & 4294901760u)));
                        float2 t_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&tz_5), "l"(*(const unsigned long long*)&tw_5));
                        y[10] = t_6.x;
                        y[11] = t_6.y;
                        float2 _f32x2_mul_a_163 = make_float2(__uint_as_float(tx[6] << 16), __uint_as_float(tx[6] & 4294901760u));
                        float2 tz_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_163), "l"(*(const unsigned long long*)&r2));
                        float2 tw_6 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[6] << 16), __uint_as_float(tw[6] & 4294901760u)));
                        float2 t_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&tz_6), "l"(*(const unsigned long long*)&tw_6));
                        y[12] = t_7.x;
                        y[13] = t_7.y;
                        float2 _f32x2_mul_a_164 = make_float2(__uint_as_float(tx[7] << 16), __uint_as_float(tx[7] & 4294901760u));
                        float2 tz_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&r2));
                        float2 tw_7 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[7] << 16), __uint_as_float(tw[7] & 4294901760u)));
                        float2 t_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&tz_7), "l"(*(const unsigned long long*)&tw_7));
                        y[14] = t_8.x;
                        y[15] = t_8.y;
                        float2 _f32x2_mul_a_165 = make_float2(__uint_as_float(tx[8] << 16), __uint_as_float(tx[8] & 4294901760u));
                        float2 tz_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&r2));
                        float2 tw_8 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[8] << 16), __uint_as_float(tw[8] & 4294901760u)));
                        float2 t_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&tz_8), "l"(*(const unsigned long long*)&tw_8));
                        y[16] = t_9.x;
                        y[17] = t_9.y;
                        float2 _f32x2_mul_a_166 = make_float2(__uint_as_float(tx[9] << 16), __uint_as_float(tx[9] & 4294901760u));
                        float2 tz_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&r2));
                        float2 tw_9 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[9] << 16), __uint_as_float(tw[9] & 4294901760u)));
                        float2 t_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&tz_9), "l"(*(const unsigned long long*)&tw_9));
                        y[18] = t_10.x;
                        y[19] = t_10.y;
                        float2 _f32x2_mul_a_167 = make_float2(__uint_as_float(tx[10] << 16), __uint_as_float(tx[10] & 4294901760u));
                        float2 tz_10;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_167), "l"(*(const unsigned long long*)&r2));
                        float2 tw_10 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[10] << 16), __uint_as_float(tw[10] & 4294901760u)));
                        float2 t_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&tz_10), "l"(*(const unsigned long long*)&tw_10));
                        y[20] = t_11.x;
                        y[21] = t_11.y;
                        float2 _f32x2_mul_a_168 = make_float2(__uint_as_float(tx[11] << 16), __uint_as_float(tx[11] & 4294901760u));
                        float2 tz_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&r2));
                        float2 tw_11 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[11] << 16), __uint_as_float(tw[11] & 4294901760u)));
                        float2 t_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&tz_11), "l"(*(const unsigned long long*)&tw_11));
                        y[22] = t_12.x;
                        y[23] = t_12.y;
                        float2 _f32x2_mul_a_169 = make_float2(__uint_as_float(tx[12] << 16), __uint_as_float(tx[12] & 4294901760u));
                        float2 tz_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_169), "l"(*(const unsigned long long*)&r2));
                        float2 tw_12 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[12] << 16), __uint_as_float(tw[12] & 4294901760u)));
                        float2 t_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&tz_12), "l"(*(const unsigned long long*)&tw_12));
                        y[24] = t_13.x;
                        y[25] = t_13.y;
                        float2 _f32x2_mul_a_170 = make_float2(__uint_as_float(tx[13] << 16), __uint_as_float(tx[13] & 4294901760u));
                        float2 tz_13;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&r2));
                        float2 tw_13 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[13] << 16), __uint_as_float(tw[13] & 4294901760u)));
                        float2 t_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&tz_13), "l"(*(const unsigned long long*)&tw_13));
                        y[26] = t_14.x;
                        y[27] = t_14.y;
                        float2 _f32x2_mul_a_171 = make_float2(__uint_as_float(tx[14] << 16), __uint_as_float(tx[14] & 4294901760u));
                        float2 tz_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_171), "l"(*(const unsigned long long*)&r2));
                        float2 tw_14 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[14] << 16), __uint_as_float(tw[14] & 4294901760u)));
                        float2 t_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&tz_14), "l"(*(const unsigned long long*)&tw_14));
                        y[28] = t_15.x;
                        y[29] = t_15.y;
                        float2 _f32x2_mul_a_172 = make_float2(__uint_as_float(tx[15] << 16), __uint_as_float(tx[15] & 4294901760u));
                        float2 tz_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&r2));
                        float2 tw_15 = add_f32x2_noftz(b2, make_float2(__uint_as_float(tw[15] << 16), __uint_as_float(tw[15] & 4294901760u)));
                        float2 t_16;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&tz_15), "l"(*(const unsigned long long*)&tw_15));
                        y[30] = t_16.x;
                        y[31] = t_16.y;
                        unsigned int y_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        long long hoff = sf_base + (long long)(intile + (unsigned int)(8192 * j));
                        float zero = 0.0f;
                        bool fodd_3 = (f & 1) != 0;
                        bool fhi_5 = (f & 2) != 0;
                        float bm[2];
                        unsigned int a_65 = 0;
                        uint32_t ab_0;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_0) : "r"(y_bf16[0]));
                        uint32_t mx_0;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_0) : "r"(a_65), "r"(ab_0));
                        a_65 = mx_0;
                        uint32_t ab_1;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_1) : "r"(y_bf16[1]));
                        uint32_t mx_1;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_1) : "r"(a_65), "r"(ab_1));
                        a_65 = mx_1;
                        uint32_t ab_2;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_2) : "r"(y_bf16[2]));
                        uint32_t mx_2;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_2) : "r"(a_65), "r"(ab_2));
                        a_65 = mx_2;
                        uint32_t ab_3;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_3) : "r"(y_bf16[3]));
                        uint32_t mx_3;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_3) : "r"(a_65), "r"(ab_3));
                        a_65 = mx_3;
                        uint32_t ab_4;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_4) : "r"(y_bf16[4]));
                        uint32_t mx_4;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_4) : "r"(a_65), "r"(ab_4));
                        a_65 = mx_4;
                        uint32_t ab_5;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_5) : "r"(y_bf16[5]));
                        uint32_t mx_5;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_5) : "r"(a_65), "r"(ab_5));
                        a_65 = mx_5;
                        uint32_t ab_6;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_6) : "r"(y_bf16[6]));
                        uint32_t mx_6;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_6) : "r"(a_65), "r"(ab_6));
                        a_65 = mx_6;
                        uint32_t ab_7;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_7) : "r"(y_bf16[7]));
                        uint32_t mx_7;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_7) : "r"(a_65), "r"(ab_7));
                        a_65 = mx_7;
                        bm[0] = fmaxf(__uint_as_float(a_65 << 16), __uint_as_float(a_65 & 4294901760u));
                        unsigned int a_66 = 0;
                        uint32_t ab_8;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_8) : "r"(y_bf16[8]));
                        uint32_t mx_8;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_8) : "r"(a_66), "r"(ab_8));
                        a_66 = mx_8;
                        uint32_t ab_9;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_9) : "r"(y_bf16[9]));
                        uint32_t mx_9;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_9) : "r"(a_66), "r"(ab_9));
                        a_66 = mx_9;
                        uint32_t ab_10;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_10) : "r"(y_bf16[10]));
                        uint32_t mx_10;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_10) : "r"(a_66), "r"(ab_10));
                        a_66 = mx_10;
                        uint32_t ab_11;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_11) : "r"(y_bf16[11]));
                        uint32_t mx_11;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_11) : "r"(a_66), "r"(ab_11));
                        a_66 = mx_11;
                        uint32_t ab_12;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_12) : "r"(y_bf16[12]));
                        uint32_t mx_12;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_12) : "r"(a_66), "r"(ab_12));
                        a_66 = mx_12;
                        uint32_t ab_13;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_13) : "r"(y_bf16[13]));
                        uint32_t mx_13;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_13) : "r"(a_66), "r"(ab_13));
                        a_66 = mx_13;
                        uint32_t ab_14;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_14) : "r"(y_bf16[14]));
                        uint32_t mx_14;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_14) : "r"(a_66), "r"(ab_14));
                        a_66 = mx_14;
                        uint32_t ab_15;
                        asm("abs.bf16x2 %0, %1;" : "=r"(ab_15) : "r"(y_bf16[15]));
                        uint32_t mx_15;
                        asm("max.bf16x2 %0, %1, %2;" : "=r"(mx_15) : "r"(a_66), "r"(ab_15));
                        a_66 = mx_15;
                        bm[1] = fmaxf(__uint_as_float(a_66 << 16), __uint_as_float(a_66 & 4294901760u));
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
                        if (fhi_5) {
                            vm0 = bm[1];
                            vm1 = bm[0];
                        }
                        unsigned int sfb[2];
                        float sfv;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv) : "f"(gso), "f"(mul_f32_rn_noftz(vm0, rcp6)));
                        uint16_t _e4m3x2_f32_6;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_6) : "f"(zero), "f"(sfv));
                        sfb[0] = (unsigned int)_e4m3x2_f32_6 & 255;
                        uint32_t e4m3x2_to_f16x2_8;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_8) : "h"((uint16_t)(_e4m3x2_f32_6)));
                        float cvt_f32_f16_9;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_9) : "h"((uint16_t)(e4m3x2_to_f16x2_8)));
                        float cvt_f32_f16_1_8;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_8) : "h"((uint16_t)(e4m3x2_to_f16x2_8 >> 16)));
                        float _rcp_0 = approx_rcp(mul_f32_rn_noftz(cvt_f32_f16_9, rcp_gs));
                        float oscale = 0.0f;
                        if (vm0 != 0.0f) {
                            oscale = _rcp_0;
                        }
                        float sfv_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfv_47) : "f"(gso), "f"(mul_f32_rn_noftz(vm1, rcp6)));
                        uint16_t _e4m3x2_f32_7;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_7) : "f"(zero), "f"(sfv_47));
                        sfb[1] = (unsigned int)_e4m3x2_f32_7 & 255;
                        uint32_t e4m3x2_to_f16x2_8_1;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(e4m3x2_to_f16x2_8_1) : "h"((uint16_t)(_e4m3x2_f32_7)));
                        float cvt_f32_f16_9_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_9_1) : "h"((uint16_t)(e4m3x2_to_f16x2_8_1)));
                        float cvt_f32_f16_1_8_1;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(cvt_f32_f16_1_8_1) : "h"((uint16_t)(e4m3x2_to_f16x2_8_1 >> 16)));
                        float _rcp_1 = approx_rcp(mul_f32_rn_noftz(cvt_f32_f16_9_1, rcp_gs));
                        float oscale_48 = 0.0f;
                        if (vm1 != 0.0f) {
                            oscale_48 = _rcp_1;
                        }
                        float os_p0 = oscale;
                        float os_p1 = oscale_48;
                        if (fhi_5) {
                            os_p0 = oscale_48;
                            os_p1 = oscale;
                        }
                        float2 os2_p0 = make_float2(os_p0, os_p0);
                        float2 os2_p1 = make_float2(os_p1, os_p1);
                        unsigned int w_49[4];
                        float2 _f32x2_mul_a_173 = make_float2(yb[0], yb[1]);
                        float2 g_26;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_72;
                        uint16_t _e2m1_pair_h_174;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_174) : "f"(g_26.x), "f"(g_26.y));
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
                        float2 g_53;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_75;
                        uint16_t _e2m1_pair_h_183;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_183) : "f"(g_53.x), "f"(g_53.y));
                        _fp4_pair_75 = (uint32_t)_e2m1_pair_h_183;
                        float2 _f32x2_mul_a_184 = make_float2(yb[10], yb[11]);
                        float2 g_54;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_76;
                        uint16_t _e2m1_pair_h_185;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_185) : "f"(g_54.x), "f"(g_54.y));
                        _fp4_pair_76 = (uint32_t)_e2m1_pair_h_185;
                        float2 _f32x2_mul_a_186 = make_float2(yb[12], yb[13]);
                        float2 g_55;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&os2_p0));
                        uint32_t _fp4_pair_77;
                        uint16_t _e2m1_pair_h_187;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_187) : "f"(g_55.x), "f"(g_55.y));
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
                        float2 g_57;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_78;
                        uint16_t _e2m1_pair_h_192;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_192) : "f"(g_57.x), "f"(g_57.y));
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
                        float2 g_63;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g_63) : "l"(*(const unsigned long long*)&_f32x2_mul_a_204), "l"(*(const unsigned long long*)&os2_p1));
                        uint32_t _fp4_pair_83;
                        uint16_t _e2m1_pair_h_205;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_205) : "f"(g_63.x), "f"(g_63.y));
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
                        if (fodd_3) {
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
                        if (fhi_5) {
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
                        unsigned int lo_3 = half & ~oddmask | _shfl_xor_0 & oddmask;
                        unsigned int hi_3 = _shfl_xor_0 & ~oddmask | half & oddmask;
                        unsigned int word_23_1 = lo_3 | hi_3 << 16;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out_sf + hoff)) + (0)) = word_23_1;
                    }
                }
                if (n > k + 1) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (lane_2 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        long long xrow = (long long)(r0 + warp_1 + 11 * (k + 1));
                        mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 12288);
                        cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)xb_off, x + (xrow * 6144), 12288, xb_full_addr + (warp_1) * 8);
                    }
                }
                _phase_xb_full ^= 1;
                _phase_rb_full ^= 1;
            }
            if (lane_2 == 0) {
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
    if (warp == 11) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("barrier.sync 0, 384;" ::: "memory");
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
                long long lo_4 = u * bid64_1 / g64_1;
                long long hi_4 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_4 + lane64;
                #pragma unroll 1
                for (long long g_27 = g0; g_27 < hi_4; g_27 += 32) {
                    long long widx = v_base_1 + 16 * g_27 >> 2;
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
                for (long long g_28 = g0_0; g_28 < hi_3_1; g_28 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_28 >> 2;
                    {
                        int4 _iv4 = make_int4(ss4[0 + 0], ss4[0 + 1], ss4[0 + 2], ss4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx_1) = _iv4;
                    }
                }
                long long u_1 = pr64 * 16 / 16;
                long long lo_4_1 = u_1 * bid64_1 / g64_1;
                long long hi_4_1 = u_1 * (bid64_1 + 1) / g64_1;
                long long g0_1 = lo_4_1 + lane64;
                #pragma unroll 1
                for (long long g_29 = g0_1; g_29 < hi_4_1; g_29 += 32) {
                    long long widx_2 = m_base_1 + 16 * g_29 >> 2;
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
