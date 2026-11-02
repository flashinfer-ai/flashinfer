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
#define SMEM_WSM_STAGE_BYTES 181632
#define SMEM_WSM_STRIDE 181632
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 181632
#define SMEM_WSM32_STRIDE 181632
#define SMEM_TOTAL 195072
#define THREADS 288
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
cake_fp4_allreduce_tp2_rmsnorm_nvfp4_bf16_pa_p2d_u2_csp(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 8 * k);
                    int ob_off = xb_off + k % 3 * 3472;
                    float rs = 1.0f;
                    mbarrier_wait(xb_full_addr + (warp_1) * 8, k & 1);
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
                    int pp = lane_2;
                    uint32_t _sysv_ld_0[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_0[0]), "=r"(_sysv_ld_0[1]), "=r"(_sysv_ld_0[2]), "=r"(_sysv_ld_0[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp)))) : "memory");
                    pv[0] = _sysv_ld_0[0];
                    pv[1] = _sysv_ld_0[1];
                    pv[2] = _sysv_ld_0[2];
                    pv[3] = _sysv_ld_0[3];
                    uint32_t _sysv_u32_0;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_0) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp >> 1))))) : "memory");
                    ps[0] = _sysv_u32_0 >> (unsigned int)(16 * (pp & 1)) & 65535;
                    int pp_0 = lane_2 + 32;
                    uint32_t _sysv_ld_1[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_1[0]), "=r"(_sysv_ld_1[1]), "=r"(_sysv_ld_1[2]), "=r"(_sysv_ld_1[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_0)))) : "memory");
                    pv[4] = _sysv_ld_1[0];
                    pv[5] = _sysv_ld_1[1];
                    pv[6] = _sysv_ld_1[2];
                    pv[7] = _sysv_ld_1[3];
                    uint32_t _sysv_u32_1;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_1) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_0 >> 1))))) : "memory");
                    ps[1] = _sysv_u32_1 >> (unsigned int)(16 * (pp_0 & 1)) & 65535;
                    int pp_1 = lane_2 + 64;
                    uint32_t _sysv_ld_2[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_2[0]), "=r"(_sysv_ld_2[1]), "=r"(_sysv_ld_2[2]), "=r"(_sysv_ld_2[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_1)))) : "memory");
                    pv[8] = _sysv_ld_2[0];
                    pv[9] = _sysv_ld_2[1];
                    pv[10] = _sysv_ld_2[2];
                    pv[11] = _sysv_ld_2[3];
                    uint32_t _sysv_u32_2;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_2) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_1 >> 1))))) : "memory");
                    ps[2] = _sysv_u32_2 >> (unsigned int)(16 * (pp_1 & 1)) & 65535;
                    int pp_2 = lane_2 + 96;
                    uint32_t _sysv_ld_3[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_3[0]), "=r"(_sysv_ld_3[1]), "=r"(_sysv_ld_3[2]), "=r"(_sysv_ld_3[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_2)))) : "memory");
                    pv[12] = _sysv_ld_3[0];
                    pv[13] = _sysv_ld_3[1];
                    pv[14] = _sysv_ld_3[2];
                    pv[15] = _sysv_ld_3[3];
                    uint32_t _sysv_u32_3;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_3) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_2 >> 1))))) : "memory");
                    ps[3] = _sysv_u32_3 >> (unsigned int)(16 * (pp_2 & 1)) & 65535;
                    int pp_3 = lane_2 + 128;
                    uint32_t _sysv_ld_4[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_4[0]), "=r"(_sysv_ld_4[1]), "=r"(_sysv_ld_4[2]), "=r"(_sysv_ld_4[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_3)))) : "memory");
                    pv[16] = _sysv_ld_4[0];
                    pv[17] = _sysv_ld_4[1];
                    pv[18] = _sysv_ld_4[2];
                    pv[19] = _sysv_ld_4[3];
                    uint32_t _sysv_u32_4;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_4) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_3 >> 1))))) : "memory");
                    ps[4] = _sysv_u32_4 >> (unsigned int)(16 * (pp_3 & 1)) & 65535;
                    int pp_4 = lane_2 + 160;
                    uint32_t _sysv_ld_5[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_ld_5[0]), "=r"(_sysv_ld_5[1]), "=r"(_sysv_ld_5[2]), "=r"(_sysv_ld_5[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp_4)))) : "memory");
                    pv[20] = _sysv_ld_5[0];
                    pv[21] = _sysv_ld_5[1];
                    pv[22] = _sysv_ld_5[2];
                    pv[23] = _sysv_ld_5[3];
                    uint32_t _sysv_u32_5;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_5) : "l"(reinterpret_cast<unsigned int*>(os_wire + (s_row + (long long)(4 * (pp_4 >> 1))))) : "memory");
                    ps[5] = _sysv_u32_5 >> (unsigned int)(16 * (pp_4 & 1)) & 65535;
                    unsigned int rsw[1];
                    uint32_t _sysv_u32_6;
                    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_u32_6) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    rsw[0] = _sysv_u32_6;
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
                    mbarrier_wait(rb_full_addr + (warp_1) * 8, kd & 1);
                    float rs_peer_f = __uint_as_float(rsw[0]);
                    float rs_a_f = ((first) ? own_rs : rs_peer_f);
                    float rs_b_f = ((first) ? rs_peer_f : own_rs);
                    int pp_5 = lane_2;
                    unsigned int own[4];
                    unsigned int own_sc[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own[0])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_5)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_5)));
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
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(a_sc)));
                    float _cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                    float _cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_0), "f"(rs_a_f));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_1), "f"(rs_a_f));
                    float2 scale0 = make_float2(_mul_0, _mul_0);
                    float2 scale1 = make_float2(_mul_1, _mul_1);
                    float _unpack_fp4_0[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[0]));
                        (&_unpack_fp4_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_0[0]), "=f"(_unpack_fp4_0[1]), "=f"(_unpack_fp4_0[2]), "=f"(_unpack_fp4_0[3]), "=f"(_unpack_fp4_0[4]), "=f"(_unpack_fp4_0[5]), "=f"(_unpack_fp4_0[6]), "=f"(_unpack_fp4_0[7]) : "r"(a[0]));
                        #endif
                    }
                    float2 a_6 = make_float2(acc[0], acc[1]);
                    float2 q = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 r = fma_f32x2_rn_noftz(q, scale0, a_6);
                    acc[0] = r.x;
                    acc[1] = r.y;
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
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[1 + 0]));
                        (&_unpack_fp4_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_1[0]), "=f"(_unpack_fp4_1[1]), "=f"(_unpack_fp4_1[2]), "=f"(_unpack_fp4_1[3]), "=f"(_unpack_fp4_1[4]), "=f"(_unpack_fp4_1[5]), "=f"(_unpack_fp4_1[6]), "=f"(_unpack_fp4_1[7]) : "r"(a[1 + 0]));
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
                    float2 a_22 = make_float2(acc[12], acc[13]);
                    float2 q_23 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 r_24 = fma_f32x2_rn_noftz(q_23, scale0, a_22);
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
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[2 + 0]));
                        (&_unpack_fp4_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_2[0]), "=f"(_unpack_fp4_2[1]), "=f"(_unpack_fp4_2[2]), "=f"(_unpack_fp4_2[3]), "=f"(_unpack_fp4_2[4]), "=f"(_unpack_fp4_2[5]), "=f"(_unpack_fp4_2[6]), "=f"(_unpack_fp4_2[7]) : "r"(a[2 + 0]));
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
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a[3 + 0]));
                        (&_unpack_fp4_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_3[0]), "=f"(_unpack_fp4_3[1]), "=f"(_unpack_fp4_3[2]), "=f"(_unpack_fp4_3[3]), "=f"(_unpack_fp4_3[4]), "=f"(_unpack_fp4_3[5]), "=f"(_unpack_fp4_3[6]), "=f"(_unpack_fp4_3[7]) : "r"(a[3 + 0]));
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
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(b_sc)));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_2), "f"(rs_b_f));
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_3), "f"(rs_b_f));
                    float2 scale0_52 = make_float2(_mul_2, _mul_2);
                    float2 scale1_53 = make_float2(_mul_3, _mul_3);
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(b[0]));
                        #endif
                    }
                    float2 a_54 = make_float2(acc[0], acc[1]);
                    float2 q_55 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 r_56 = fma_f32x2_rn_noftz(q_55, scale0_52, a_54);
                    acc[0] = r_56.x;
                    acc[1] = r_56.y;
                    float2 a_57 = make_float2(acc[2], acc[3]);
                    float2 q_58 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 r_59 = fma_f32x2_rn_noftz(q_58, scale0_52, a_57);
                    acc[2] = r_59.x;
                    acc[3] = r_59.y;
                    float2 a_60 = make_float2(acc[4], acc[5]);
                    float2 q_61 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 r_62 = fma_f32x2_rn_noftz(q_61, scale0_52, a_60);
                    acc[4] = r_62.x;
                    acc[5] = r_62.y;
                    float2 a_63 = make_float2(acc[6], acc[7]);
                    float2 q_64 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 r_65 = fma_f32x2_rn_noftz(q_64, scale0_52, a_63);
                    acc[6] = r_65.x;
                    acc[7] = r_65.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(b[1 + 0]));
                        #endif
                    }
                    float2 a_66 = make_float2(acc[8], acc[9]);
                    float2 q_67 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 r_68 = fma_f32x2_rn_noftz(q_67, scale0_52, a_66);
                    acc[8] = r_68.x;
                    acc[9] = r_68.y;
                    float2 a_69 = make_float2(acc[10], acc[11]);
                    float2 q_70 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 r_71 = fma_f32x2_rn_noftz(q_70, scale0_52, a_69);
                    acc[10] = r_71.x;
                    acc[11] = r_71.y;
                    float2 a_72 = make_float2(acc[12], acc[13]);
                    float2 q_73 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 r_74 = fma_f32x2_rn_noftz(q_73, scale0_52, a_72);
                    acc[12] = r_74.x;
                    acc[13] = r_74.y;
                    float2 a_75 = make_float2(acc[14], acc[15]);
                    float2 q_76 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 r_77 = fma_f32x2_rn_noftz(q_76, scale0_52, a_75);
                    acc[14] = r_77.x;
                    acc[15] = r_77.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(b[2 + 0]));
                        #endif
                    }
                    float2 a_78 = make_float2(acc[16], acc[17]);
                    float2 q_79 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 r_80 = fma_f32x2_rn_noftz(q_79, scale1_53, a_78);
                    acc[16] = r_80.x;
                    acc[17] = r_80.y;
                    float2 a_81 = make_float2(acc[18], acc[19]);
                    float2 q_82 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 r_83 = fma_f32x2_rn_noftz(q_82, scale1_53, a_81);
                    acc[18] = r_83.x;
                    acc[19] = r_83.y;
                    float2 a_84 = make_float2(acc[20], acc[21]);
                    float2 q_85 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 r_86 = fma_f32x2_rn_noftz(q_85, scale1_53, a_84);
                    acc[20] = r_86.x;
                    acc[21] = r_86.y;
                    float2 a_87 = make_float2(acc[22], acc[23]);
                    float2 q_88 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 r_89 = fma_f32x2_rn_noftz(q_88, scale1_53, a_87);
                    acc[22] = r_89.x;
                    acc[23] = r_89.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(b[3 + 0]));
                        #endif
                    }
                    float2 a_90 = make_float2(acc[24], acc[25]);
                    float2 q_91 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 r_92 = fma_f32x2_rn_noftz(q_91, scale1_53, a_90);
                    acc[24] = r_92.x;
                    acc[25] = r_92.y;
                    float2 a_93 = make_float2(acc[26], acc[27]);
                    float2 q_94 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 r_95 = fma_f32x2_rn_noftz(q_94, scale1_53, a_93);
                    acc[26] = r_95.x;
                    acc[27] = r_95.y;
                    float2 a_96 = make_float2(acc[28], acc[29]);
                    float2 q_97 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 r_98 = fma_f32x2_rn_noftz(q_97, scale1_53, a_96);
                    acc[28] = r_98.x;
                    acc[29] = r_98.y;
                    float2 a_99 = make_float2(acc[30], acc[31]);
                    float2 q_100 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 r_101 = fma_f32x2_rn_noftz(q_100, scale1_53, a_99);
                    acc[30] = r_101.x;
                    acc[31] = r_101.y;
                    unsigned int t[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[0])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[4])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[8])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t[12])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 3))));
                    bool fodd = (f & 1) != 0;
                    bool fhi = (f & 2) != 0;
                    unsigned int s[16];
                    s[0] = t[0];
                    s[1] = t[1];
                    s[2] = t[2];
                    s[3] = t[3];
                    s[4] = t[4];
                    s[5] = t[5];
                    s[6] = t[6];
                    s[7] = t[7];
                    s[8] = t[8];
                    s[9] = t[9];
                    s[10] = t[10];
                    s[11] = t[11];
                    s[12] = t[12];
                    s[13] = t[13];
                    s[14] = t[14];
                    s[15] = t[15];
                    if (fodd) {
                        s[0] = t[4];
                        s[4] = t[0];
                        s[8] = t[12];
                        s[12] = t[8];
                        s[1] = t[5];
                        s[5] = t[1];
                        s[9] = t[13];
                        s[13] = t[9];
                        s[2] = t[6];
                        s[6] = t[2];
                        s[10] = t[14];
                        s[14] = t[10];
                        s[3] = t[7];
                        s[7] = t[3];
                        s[11] = t[15];
                        s[15] = t[11];
                    }
                    unsigned int r_102[16];
                    r_102[0] = s[0];
                    r_102[1] = s[1];
                    r_102[2] = s[2];
                    r_102[3] = s[3];
                    r_102[4] = s[4];
                    r_102[5] = s[5];
                    r_102[6] = s[6];
                    r_102[7] = s[7];
                    r_102[8] = s[8];
                    r_102[9] = s[9];
                    r_102[10] = s[10];
                    r_102[11] = s[11];
                    r_102[12] = s[12];
                    r_102[13] = s[13];
                    r_102[14] = s[14];
                    r_102[15] = s[15];
                    if (fhi) {
                        r_102[0] = s[8];
                        r_102[4] = s[12];
                        r_102[8] = s[0];
                        r_102[12] = s[4];
                        r_102[1] = s[9];
                        r_102[5] = s[13];
                        r_102[9] = s[1];
                        r_102[13] = s[5];
                        r_102[2] = s[10];
                        r_102[6] = s[14];
                        r_102[10] = s[2];
                        r_102[14] = s[6];
                        r_102[3] = s[11];
                        r_102[7] = s[15];
                        r_102[11] = s[3];
                        r_102[15] = s[7];
                    }
                    float2 a2 = make_float2(acc[0], acc[1]);
                    float2 z = add_f32x2_noftz(a2, make_float2(__uint_as_float(r_102[0] << 16), __uint_as_float(r_102[0] & 4294901760u)));
                    acc[0] = z.x;
                    acc[1] = z.y;
                    float2 ss2 = fma_f32x2_rn_noftz(z, z, make_float2(ssx, ssy));
                    ssx = ss2.x;
                    ssy = ss2.y;
                    float2 a2_103 = make_float2(acc[2], acc[3]);
                    float2 z_104 = add_f32x2_noftz(a2_103, make_float2(__uint_as_float(r_102[1] << 16), __uint_as_float(r_102[1] & 4294901760u)));
                    acc[2] = z_104.x;
                    acc[3] = z_104.y;
                    float2 ss2_105 = fma_f32x2_rn_noftz(z_104, z_104, make_float2(ssx, ssy));
                    ssx = ss2_105.x;
                    ssy = ss2_105.y;
                    float2 a2_106 = make_float2(acc[4], acc[5]);
                    float2 z_107 = add_f32x2_noftz(a2_106, make_float2(__uint_as_float(r_102[2] << 16), __uint_as_float(r_102[2] & 4294901760u)));
                    acc[4] = z_107.x;
                    acc[5] = z_107.y;
                    float2 ss2_108 = fma_f32x2_rn_noftz(z_107, z_107, make_float2(ssx, ssy));
                    ssx = ss2_108.x;
                    ssy = ss2_108.y;
                    float2 a2_109 = make_float2(acc[6], acc[7]);
                    float2 z_110 = add_f32x2_noftz(a2_109, make_float2(__uint_as_float(r_102[3] << 16), __uint_as_float(r_102[3] & 4294901760u)));
                    acc[6] = z_110.x;
                    acc[7] = z_110.y;
                    float2 ss2_111 = fma_f32x2_rn_noftz(z_110, z_110, make_float2(ssx, ssy));
                    ssx = ss2_111.x;
                    ssy = ss2_111.y;
                    float2 a2_112 = make_float2(acc[8], acc[9]);
                    float2 z_113 = add_f32x2_noftz(a2_112, make_float2(__uint_as_float(r_102[4] << 16), __uint_as_float(r_102[4] & 4294901760u)));
                    acc[8] = z_113.x;
                    acc[9] = z_113.y;
                    float2 ss2_114 = fma_f32x2_rn_noftz(z_113, z_113, make_float2(ssx, ssy));
                    ssx = ss2_114.x;
                    ssy = ss2_114.y;
                    float2 a2_115 = make_float2(acc[10], acc[11]);
                    float2 z_116 = add_f32x2_noftz(a2_115, make_float2(__uint_as_float(r_102[5] << 16), __uint_as_float(r_102[5] & 4294901760u)));
                    acc[10] = z_116.x;
                    acc[11] = z_116.y;
                    float2 ss2_117 = fma_f32x2_rn_noftz(z_116, z_116, make_float2(ssx, ssy));
                    ssx = ss2_117.x;
                    ssy = ss2_117.y;
                    float2 a2_118 = make_float2(acc[12], acc[13]);
                    float2 z_119 = add_f32x2_noftz(a2_118, make_float2(__uint_as_float(r_102[6] << 16), __uint_as_float(r_102[6] & 4294901760u)));
                    acc[12] = z_119.x;
                    acc[13] = z_119.y;
                    float2 ss2_120 = fma_f32x2_rn_noftz(z_119, z_119, make_float2(ssx, ssy));
                    ssx = ss2_120.x;
                    ssy = ss2_120.y;
                    float2 a2_121 = make_float2(acc[14], acc[15]);
                    float2 z_122 = add_f32x2_noftz(a2_121, make_float2(__uint_as_float(r_102[7] << 16), __uint_as_float(r_102[7] & 4294901760u)));
                    acc[14] = z_122.x;
                    acc[15] = z_122.y;
                    float2 ss2_123 = fma_f32x2_rn_noftz(z_122, z_122, make_float2(ssx, ssy));
                    ssx = ss2_123.x;
                    ssy = ss2_123.y;
                    float2 a2_124 = make_float2(acc[16], acc[17]);
                    float2 z_125 = add_f32x2_noftz(a2_124, make_float2(__uint_as_float(r_102[8] << 16), __uint_as_float(r_102[8] & 4294901760u)));
                    acc[16] = z_125.x;
                    acc[17] = z_125.y;
                    float2 ss2_126 = fma_f32x2_rn_noftz(z_125, z_125, make_float2(ssx, ssy));
                    ssx = ss2_126.x;
                    ssy = ss2_126.y;
                    float2 a2_127 = make_float2(acc[18], acc[19]);
                    float2 z_128 = add_f32x2_noftz(a2_127, make_float2(__uint_as_float(r_102[9] << 16), __uint_as_float(r_102[9] & 4294901760u)));
                    acc[18] = z_128.x;
                    acc[19] = z_128.y;
                    float2 ss2_129 = fma_f32x2_rn_noftz(z_128, z_128, make_float2(ssx, ssy));
                    ssx = ss2_129.x;
                    ssy = ss2_129.y;
                    float2 a2_130 = make_float2(acc[20], acc[21]);
                    float2 z_131 = add_f32x2_noftz(a2_130, make_float2(__uint_as_float(r_102[10] << 16), __uint_as_float(r_102[10] & 4294901760u)));
                    acc[20] = z_131.x;
                    acc[21] = z_131.y;
                    float2 ss2_132 = fma_f32x2_rn_noftz(z_131, z_131, make_float2(ssx, ssy));
                    ssx = ss2_132.x;
                    ssy = ss2_132.y;
                    float2 a2_133 = make_float2(acc[22], acc[23]);
                    float2 z_134 = add_f32x2_noftz(a2_133, make_float2(__uint_as_float(r_102[11] << 16), __uint_as_float(r_102[11] & 4294901760u)));
                    acc[22] = z_134.x;
                    acc[23] = z_134.y;
                    float2 ss2_135 = fma_f32x2_rn_noftz(z_134, z_134, make_float2(ssx, ssy));
                    ssx = ss2_135.x;
                    ssy = ss2_135.y;
                    float2 a2_136 = make_float2(acc[24], acc[25]);
                    float2 z_137 = add_f32x2_noftz(a2_136, make_float2(__uint_as_float(r_102[12] << 16), __uint_as_float(r_102[12] & 4294901760u)));
                    acc[24] = z_137.x;
                    acc[25] = z_137.y;
                    float2 ss2_138 = fma_f32x2_rn_noftz(z_137, z_137, make_float2(ssx, ssy));
                    ssx = ss2_138.x;
                    ssy = ss2_138.y;
                    float2 a2_139 = make_float2(acc[26], acc[27]);
                    float2 z_140 = add_f32x2_noftz(a2_139, make_float2(__uint_as_float(r_102[13] << 16), __uint_as_float(r_102[13] & 4294901760u)));
                    acc[26] = z_140.x;
                    acc[27] = z_140.y;
                    float2 ss2_141 = fma_f32x2_rn_noftz(z_140, z_140, make_float2(ssx, ssy));
                    ssx = ss2_141.x;
                    ssy = ss2_141.y;
                    float2 a2_142 = make_float2(acc[28], acc[29]);
                    float2 z_143 = add_f32x2_noftz(a2_142, make_float2(__uint_as_float(r_102[14] << 16), __uint_as_float(r_102[14] & 4294901760u)));
                    acc[28] = z_143.x;
                    acc[29] = z_143.y;
                    float2 ss2_144 = fma_f32x2_rn_noftz(z_143, z_143, make_float2(ssx, ssy));
                    ssx = ss2_144.x;
                    ssy = ss2_144.y;
                    float2 a2_145 = make_float2(acc[30], acc[31]);
                    float2 z_146 = add_f32x2_noftz(a2_145, make_float2(__uint_as_float(r_102[15] << 16), __uint_as_float(r_102[15] & 4294901760u)));
                    acc[30] = z_146.x;
                    acc[31] = z_146.y;
                    float2 ss2_147 = fma_f32x2_rn_noftz(z_146, z_146, make_float2(ssx, ssy));
                    ssx = ss2_147.x;
                    ssy = ss2_147.y;
                    unsigned int acc_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_148 = (f & 1) != 0;
                    bool fhi_149 = (f & 2) != 0;
                    unsigned int s_150[16];
                    s_150[0] = acc_bf16[0];
                    s_150[1] = acc_bf16[1];
                    s_150[2] = acc_bf16[2];
                    s_150[3] = acc_bf16[3];
                    s_150[4] = acc_bf16[4];
                    s_150[5] = acc_bf16[5];
                    s_150[6] = acc_bf16[6];
                    s_150[7] = acc_bf16[7];
                    s_150[8] = acc_bf16[8];
                    s_150[9] = acc_bf16[9];
                    s_150[10] = acc_bf16[10];
                    s_150[11] = acc_bf16[11];
                    s_150[12] = acc_bf16[12];
                    s_150[13] = acc_bf16[13];
                    s_150[14] = acc_bf16[14];
                    s_150[15] = acc_bf16[15];
                    if (fodd_148) {
                        s_150[0] = acc_bf16[4];
                        s_150[4] = acc_bf16[0];
                        s_150[8] = acc_bf16[12];
                        s_150[12] = acc_bf16[8];
                        s_150[1] = acc_bf16[5];
                        s_150[5] = acc_bf16[1];
                        s_150[9] = acc_bf16[13];
                        s_150[13] = acc_bf16[9];
                        s_150[2] = acc_bf16[6];
                        s_150[6] = acc_bf16[2];
                        s_150[10] = acc_bf16[14];
                        s_150[14] = acc_bf16[10];
                        s_150[3] = acc_bf16[7];
                        s_150[7] = acc_bf16[3];
                        s_150[11] = acc_bf16[15];
                        s_150[15] = acc_bf16[11];
                    }
                    unsigned int r_151[16];
                    r_151[0] = s_150[0];
                    r_151[1] = s_150[1];
                    r_151[2] = s_150[2];
                    r_151[3] = s_150[3];
                    r_151[4] = s_150[4];
                    r_151[5] = s_150[5];
                    r_151[6] = s_150[6];
                    r_151[7] = s_150[7];
                    r_151[8] = s_150[8];
                    r_151[9] = s_150[9];
                    r_151[10] = s_150[10];
                    r_151[11] = s_150[11];
                    r_151[12] = s_150[12];
                    r_151[13] = s_150[13];
                    r_151[14] = s_150[14];
                    r_151[15] = s_150[15];
                    if (fhi_149) {
                        r_151[0] = s_150[8];
                        r_151[4] = s_150[12];
                        r_151[8] = s_150[0];
                        r_151[12] = s_150[4];
                        r_151[1] = s_150[9];
                        r_151[5] = s_150[13];
                        r_151[9] = s_150[1];
                        r_151[13] = s_150[5];
                        r_151[2] = s_150[10];
                        r_151[6] = s_150[14];
                        r_151[10] = s_150[2];
                        r_151[14] = s_150[6];
                        r_151[3] = s_150[11];
                        r_151[7] = s_150[15];
                        r_151[11] = s_150[3];
                        r_151[15] = s_150[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 0))), "r"(r_151[0]), "r"(r_151[1]), "r"(r_151[2]), "r"(r_151[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 1))), "r"(r_151[4]), "r"(r_151[5]), "r"(r_151[6]), "r"(r_151[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 2))), "r"(r_151[8]), "r"(r_151[9]), "r"(r_151[10]), "r"(r_151[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 3))), "r"(r_151[12]), "r"(r_151[13]), "r"(r_151[14]), "r"(r_151[15]) : "memory");
                    int pp_152 = lane_2 + 32;
                    unsigned int own_153[4];
                    unsigned int own_sc_154[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_153[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_153[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_153[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_153[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_152)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_154[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_152)));
                    unsigned int a_155[4];
                    unsigned int b_156[4];
                    a_155[0] = ((first) ? own_153[0] : pv[4]);
                    b_156[0] = ((first) ? pv[4] : own_153[0]);
                    a_155[1] = ((first) ? own_153[1] : pv[5]);
                    b_156[1] = ((first) ? pv[5] : own_153[1]);
                    a_155[2] = ((first) ? own_153[2] : pv[6]);
                    b_156[2] = ((first) ? pv[6] : own_153[2]);
                    a_155[3] = ((first) ? own_153[3] : pv[7]);
                    b_156[3] = ((first) ? pv[7] : own_153[3]);
                    unsigned int a_sc_157 = ((first) ? own_sc_154[0] : ps[1]);
                    unsigned int b_sc_158 = ((first) ? ps[1] : own_sc_154[0]);
                    float acc_159[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(a_sc_157)));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_4), "f"(rs_a_f));
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_5), "f"(rs_a_f));
                    float2 scale0_160 = make_float2(_mul_4, _mul_4);
                    float2 scale1_161 = make_float2(_mul_5, _mul_5);
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_155[0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(a_155[0]));
                        #endif
                    }
                    float2 a_162 = make_float2(acc_159[0], acc_159[1]);
                    float2 q_163 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 r_164 = fma_f32x2_rn_noftz(q_163, scale0_160, a_162);
                    acc_159[0] = r_164.x;
                    acc_159[1] = r_164.y;
                    float2 a_165 = make_float2(acc_159[2], acc_159[3]);
                    float2 q_166 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 r_167 = fma_f32x2_rn_noftz(q_166, scale0_160, a_165);
                    acc_159[2] = r_167.x;
                    acc_159[3] = r_167.y;
                    float2 a_168 = make_float2(acc_159[4], acc_159[5]);
                    float2 q_169 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 r_170 = fma_f32x2_rn_noftz(q_169, scale0_160, a_168);
                    acc_159[4] = r_170.x;
                    acc_159[5] = r_170.y;
                    float2 a_171 = make_float2(acc_159[6], acc_159[7]);
                    float2 q_172 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 r_173 = fma_f32x2_rn_noftz(q_172, scale0_160, a_171);
                    acc_159[6] = r_173.x;
                    acc_159[7] = r_173.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_155[1 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(a_155[1 + 0]));
                        #endif
                    }
                    float2 a_174 = make_float2(acc_159[8], acc_159[9]);
                    float2 q_175 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 r_176 = fma_f32x2_rn_noftz(q_175, scale0_160, a_174);
                    acc_159[8] = r_176.x;
                    acc_159[9] = r_176.y;
                    float2 a_177 = make_float2(acc_159[10], acc_159[11]);
                    float2 q_178 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 r_179 = fma_f32x2_rn_noftz(q_178, scale0_160, a_177);
                    acc_159[10] = r_179.x;
                    acc_159[11] = r_179.y;
                    float2 a_180 = make_float2(acc_159[12], acc_159[13]);
                    float2 q_181 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 r_182 = fma_f32x2_rn_noftz(q_181, scale0_160, a_180);
                    acc_159[12] = r_182.x;
                    acc_159[13] = r_182.y;
                    float2 a_183 = make_float2(acc_159[14], acc_159[15]);
                    float2 q_184 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 r_185 = fma_f32x2_rn_noftz(q_184, scale0_160, a_183);
                    acc_159[14] = r_185.x;
                    acc_159[15] = r_185.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_155[2 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(a_155[2 + 0]));
                        #endif
                    }
                    float2 a_186 = make_float2(acc_159[16], acc_159[17]);
                    float2 q_187 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 r_188 = fma_f32x2_rn_noftz(q_187, scale1_161, a_186);
                    acc_159[16] = r_188.x;
                    acc_159[17] = r_188.y;
                    float2 a_189 = make_float2(acc_159[18], acc_159[19]);
                    float2 q_190 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 r_191 = fma_f32x2_rn_noftz(q_190, scale1_161, a_189);
                    acc_159[18] = r_191.x;
                    acc_159[19] = r_191.y;
                    float2 a_192 = make_float2(acc_159[20], acc_159[21]);
                    float2 q_193 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 r_194 = fma_f32x2_rn_noftz(q_193, scale1_161, a_192);
                    acc_159[20] = r_194.x;
                    acc_159[21] = r_194.y;
                    float2 a_195 = make_float2(acc_159[22], acc_159[23]);
                    float2 q_196 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 r_197 = fma_f32x2_rn_noftz(q_196, scale1_161, a_195);
                    acc_159[22] = r_197.x;
                    acc_159[23] = r_197.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_155[3 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(a_155[3 + 0]));
                        #endif
                    }
                    float2 a_198 = make_float2(acc_159[24], acc_159[25]);
                    float2 q_199 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 r_200 = fma_f32x2_rn_noftz(q_199, scale1_161, a_198);
                    acc_159[24] = r_200.x;
                    acc_159[25] = r_200.y;
                    float2 a_201 = make_float2(acc_159[26], acc_159[27]);
                    float2 q_202 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 r_203 = fma_f32x2_rn_noftz(q_202, scale1_161, a_201);
                    acc_159[26] = r_203.x;
                    acc_159[27] = r_203.y;
                    float2 a_204 = make_float2(acc_159[28], acc_159[29]);
                    float2 q_205 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 r_206 = fma_f32x2_rn_noftz(q_205, scale1_161, a_204);
                    acc_159[28] = r_206.x;
                    acc_159[29] = r_206.y;
                    float2 a_207 = make_float2(acc_159[30], acc_159[31]);
                    float2 q_208 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 r_209 = fma_f32x2_rn_noftz(q_208, scale1_161, a_207);
                    acc_159[30] = r_209.x;
                    acc_159[31] = r_209.y;
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(b_sc_158)));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(rs_b_f));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(rs_b_f));
                    float2 scale0_210 = make_float2(_mul_6, _mul_6);
                    float2 scale1_211 = make_float2(_mul_7, _mul_7);
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_156[0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(b_156[0]));
                        #endif
                    }
                    float2 a_212 = make_float2(acc_159[0], acc_159[1]);
                    float2 q_213 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 r_214 = fma_f32x2_rn_noftz(q_213, scale0_210, a_212);
                    acc_159[0] = r_214.x;
                    acc_159[1] = r_214.y;
                    float2 a_215 = make_float2(acc_159[2], acc_159[3]);
                    float2 q_216 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 r_217 = fma_f32x2_rn_noftz(q_216, scale0_210, a_215);
                    acc_159[2] = r_217.x;
                    acc_159[3] = r_217.y;
                    float2 a_218 = make_float2(acc_159[4], acc_159[5]);
                    float2 q_219 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 r_220 = fma_f32x2_rn_noftz(q_219, scale0_210, a_218);
                    acc_159[4] = r_220.x;
                    acc_159[5] = r_220.y;
                    float2 a_221 = make_float2(acc_159[6], acc_159[7]);
                    float2 q_222 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 r_223 = fma_f32x2_rn_noftz(q_222, scale0_210, a_221);
                    acc_159[6] = r_223.x;
                    acc_159[7] = r_223.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_156[1 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(b_156[1 + 0]));
                        #endif
                    }
                    float2 a_224 = make_float2(acc_159[8], acc_159[9]);
                    float2 q_225 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 r_226 = fma_f32x2_rn_noftz(q_225, scale0_210, a_224);
                    acc_159[8] = r_226.x;
                    acc_159[9] = r_226.y;
                    float2 a_227 = make_float2(acc_159[10], acc_159[11]);
                    float2 q_228 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 r_229 = fma_f32x2_rn_noftz(q_228, scale0_210, a_227);
                    acc_159[10] = r_229.x;
                    acc_159[11] = r_229.y;
                    float2 a_230 = make_float2(acc_159[12], acc_159[13]);
                    float2 q_231 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 r_232 = fma_f32x2_rn_noftz(q_231, scale0_210, a_230);
                    acc_159[12] = r_232.x;
                    acc_159[13] = r_232.y;
                    float2 a_233 = make_float2(acc_159[14], acc_159[15]);
                    float2 q_234 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 r_235 = fma_f32x2_rn_noftz(q_234, scale0_210, a_233);
                    acc_159[14] = r_235.x;
                    acc_159[15] = r_235.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_156[2 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(b_156[2 + 0]));
                        #endif
                    }
                    float2 a_236 = make_float2(acc_159[16], acc_159[17]);
                    float2 q_237 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 r_238 = fma_f32x2_rn_noftz(q_237, scale1_211, a_236);
                    acc_159[16] = r_238.x;
                    acc_159[17] = r_238.y;
                    float2 a_239 = make_float2(acc_159[18], acc_159[19]);
                    float2 q_240 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 r_241 = fma_f32x2_rn_noftz(q_240, scale1_211, a_239);
                    acc_159[18] = r_241.x;
                    acc_159[19] = r_241.y;
                    float2 a_242 = make_float2(acc_159[20], acc_159[21]);
                    float2 q_243 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 r_244 = fma_f32x2_rn_noftz(q_243, scale1_211, a_242);
                    acc_159[20] = r_244.x;
                    acc_159[21] = r_244.y;
                    float2 a_245 = make_float2(acc_159[22], acc_159[23]);
                    float2 q_246 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 r_247 = fma_f32x2_rn_noftz(q_246, scale1_211, a_245);
                    acc_159[22] = r_247.x;
                    acc_159[23] = r_247.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_156[3 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(b_156[3 + 0]));
                        #endif
                    }
                    float2 a_248 = make_float2(acc_159[24], acc_159[25]);
                    float2 q_249 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 r_250 = fma_f32x2_rn_noftz(q_249, scale1_211, a_248);
                    acc_159[24] = r_250.x;
                    acc_159[25] = r_250.y;
                    float2 a_251 = make_float2(acc_159[26], acc_159[27]);
                    float2 q_252 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 r_253 = fma_f32x2_rn_noftz(q_252, scale1_211, a_251);
                    acc_159[26] = r_253.x;
                    acc_159[27] = r_253.y;
                    float2 a_254 = make_float2(acc_159[28], acc_159[29]);
                    float2 q_255 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 r_256 = fma_f32x2_rn_noftz(q_255, scale1_211, a_254);
                    acc_159[28] = r_256.x;
                    acc_159[29] = r_256.y;
                    float2 a_257 = make_float2(acc_159[30], acc_159[31]);
                    float2 q_258 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 r_259 = fma_f32x2_rn_noftz(q_258, scale1_211, a_257);
                    acc_159[30] = r_259.x;
                    acc_159[31] = r_259.y;
                    unsigned int t_260[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_260[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_260[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_260[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_260[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_260[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 3))));
                    bool fodd_261 = (f & 1) != 0;
                    bool fhi_262 = (f & 2) != 0;
                    unsigned int s_263[16];
                    s_263[0] = t_260[0];
                    s_263[1] = t_260[1];
                    s_263[2] = t_260[2];
                    s_263[3] = t_260[3];
                    s_263[4] = t_260[4];
                    s_263[5] = t_260[5];
                    s_263[6] = t_260[6];
                    s_263[7] = t_260[7];
                    s_263[8] = t_260[8];
                    s_263[9] = t_260[9];
                    s_263[10] = t_260[10];
                    s_263[11] = t_260[11];
                    s_263[12] = t_260[12];
                    s_263[13] = t_260[13];
                    s_263[14] = t_260[14];
                    s_263[15] = t_260[15];
                    if (fodd_261) {
                        s_263[0] = t_260[4];
                        s_263[4] = t_260[0];
                        s_263[8] = t_260[12];
                        s_263[12] = t_260[8];
                        s_263[1] = t_260[5];
                        s_263[5] = t_260[1];
                        s_263[9] = t_260[13];
                        s_263[13] = t_260[9];
                        s_263[2] = t_260[6];
                        s_263[6] = t_260[2];
                        s_263[10] = t_260[14];
                        s_263[14] = t_260[10];
                        s_263[3] = t_260[7];
                        s_263[7] = t_260[3];
                        s_263[11] = t_260[15];
                        s_263[15] = t_260[11];
                    }
                    unsigned int r_264[16];
                    r_264[0] = s_263[0];
                    r_264[1] = s_263[1];
                    r_264[2] = s_263[2];
                    r_264[3] = s_263[3];
                    r_264[4] = s_263[4];
                    r_264[5] = s_263[5];
                    r_264[6] = s_263[6];
                    r_264[7] = s_263[7];
                    r_264[8] = s_263[8];
                    r_264[9] = s_263[9];
                    r_264[10] = s_263[10];
                    r_264[11] = s_263[11];
                    r_264[12] = s_263[12];
                    r_264[13] = s_263[13];
                    r_264[14] = s_263[14];
                    r_264[15] = s_263[15];
                    if (fhi_262) {
                        r_264[0] = s_263[8];
                        r_264[4] = s_263[12];
                        r_264[8] = s_263[0];
                        r_264[12] = s_263[4];
                        r_264[1] = s_263[9];
                        r_264[5] = s_263[13];
                        r_264[9] = s_263[1];
                        r_264[13] = s_263[5];
                        r_264[2] = s_263[10];
                        r_264[6] = s_263[14];
                        r_264[10] = s_263[2];
                        r_264[14] = s_263[6];
                        r_264[3] = s_263[11];
                        r_264[7] = s_263[15];
                        r_264[11] = s_263[3];
                        r_264[15] = s_263[7];
                    }
                    float2 a2_265 = make_float2(acc_159[0], acc_159[1]);
                    float2 z_266 = add_f32x2_noftz(a2_265, make_float2(__uint_as_float(r_264[0] << 16), __uint_as_float(r_264[0] & 4294901760u)));
                    acc_159[0] = z_266.x;
                    acc_159[1] = z_266.y;
                    float2 ss2_267 = fma_f32x2_rn_noftz(z_266, z_266, make_float2(ssx, ssy));
                    ssx = ss2_267.x;
                    ssy = ss2_267.y;
                    float2 a2_268 = make_float2(acc_159[2], acc_159[3]);
                    float2 z_269 = add_f32x2_noftz(a2_268, make_float2(__uint_as_float(r_264[1] << 16), __uint_as_float(r_264[1] & 4294901760u)));
                    acc_159[2] = z_269.x;
                    acc_159[3] = z_269.y;
                    float2 ss2_270 = fma_f32x2_rn_noftz(z_269, z_269, make_float2(ssx, ssy));
                    ssx = ss2_270.x;
                    ssy = ss2_270.y;
                    float2 a2_271 = make_float2(acc_159[4], acc_159[5]);
                    float2 z_272 = add_f32x2_noftz(a2_271, make_float2(__uint_as_float(r_264[2] << 16), __uint_as_float(r_264[2] & 4294901760u)));
                    acc_159[4] = z_272.x;
                    acc_159[5] = z_272.y;
                    float2 ss2_273 = fma_f32x2_rn_noftz(z_272, z_272, make_float2(ssx, ssy));
                    ssx = ss2_273.x;
                    ssy = ss2_273.y;
                    float2 a2_274 = make_float2(acc_159[6], acc_159[7]);
                    float2 z_275 = add_f32x2_noftz(a2_274, make_float2(__uint_as_float(r_264[3] << 16), __uint_as_float(r_264[3] & 4294901760u)));
                    acc_159[6] = z_275.x;
                    acc_159[7] = z_275.y;
                    float2 ss2_276 = fma_f32x2_rn_noftz(z_275, z_275, make_float2(ssx, ssy));
                    ssx = ss2_276.x;
                    ssy = ss2_276.y;
                    float2 a2_277 = make_float2(acc_159[8], acc_159[9]);
                    float2 z_278 = add_f32x2_noftz(a2_277, make_float2(__uint_as_float(r_264[4] << 16), __uint_as_float(r_264[4] & 4294901760u)));
                    acc_159[8] = z_278.x;
                    acc_159[9] = z_278.y;
                    float2 ss2_279 = fma_f32x2_rn_noftz(z_278, z_278, make_float2(ssx, ssy));
                    ssx = ss2_279.x;
                    ssy = ss2_279.y;
                    float2 a2_280 = make_float2(acc_159[10], acc_159[11]);
                    float2 z_281 = add_f32x2_noftz(a2_280, make_float2(__uint_as_float(r_264[5] << 16), __uint_as_float(r_264[5] & 4294901760u)));
                    acc_159[10] = z_281.x;
                    acc_159[11] = z_281.y;
                    float2 ss2_282 = fma_f32x2_rn_noftz(z_281, z_281, make_float2(ssx, ssy));
                    ssx = ss2_282.x;
                    ssy = ss2_282.y;
                    float2 a2_283 = make_float2(acc_159[12], acc_159[13]);
                    float2 z_284 = add_f32x2_noftz(a2_283, make_float2(__uint_as_float(r_264[6] << 16), __uint_as_float(r_264[6] & 4294901760u)));
                    acc_159[12] = z_284.x;
                    acc_159[13] = z_284.y;
                    float2 ss2_285 = fma_f32x2_rn_noftz(z_284, z_284, make_float2(ssx, ssy));
                    ssx = ss2_285.x;
                    ssy = ss2_285.y;
                    float2 a2_286 = make_float2(acc_159[14], acc_159[15]);
                    float2 z_287 = add_f32x2_noftz(a2_286, make_float2(__uint_as_float(r_264[7] << 16), __uint_as_float(r_264[7] & 4294901760u)));
                    acc_159[14] = z_287.x;
                    acc_159[15] = z_287.y;
                    float2 ss2_288 = fma_f32x2_rn_noftz(z_287, z_287, make_float2(ssx, ssy));
                    ssx = ss2_288.x;
                    ssy = ss2_288.y;
                    float2 a2_289 = make_float2(acc_159[16], acc_159[17]);
                    float2 z_290 = add_f32x2_noftz(a2_289, make_float2(__uint_as_float(r_264[8] << 16), __uint_as_float(r_264[8] & 4294901760u)));
                    acc_159[16] = z_290.x;
                    acc_159[17] = z_290.y;
                    float2 ss2_291 = fma_f32x2_rn_noftz(z_290, z_290, make_float2(ssx, ssy));
                    ssx = ss2_291.x;
                    ssy = ss2_291.y;
                    float2 a2_292 = make_float2(acc_159[18], acc_159[19]);
                    float2 z_293 = add_f32x2_noftz(a2_292, make_float2(__uint_as_float(r_264[9] << 16), __uint_as_float(r_264[9] & 4294901760u)));
                    acc_159[18] = z_293.x;
                    acc_159[19] = z_293.y;
                    float2 ss2_294 = fma_f32x2_rn_noftz(z_293, z_293, make_float2(ssx, ssy));
                    ssx = ss2_294.x;
                    ssy = ss2_294.y;
                    float2 a2_295 = make_float2(acc_159[20], acc_159[21]);
                    float2 z_296 = add_f32x2_noftz(a2_295, make_float2(__uint_as_float(r_264[10] << 16), __uint_as_float(r_264[10] & 4294901760u)));
                    acc_159[20] = z_296.x;
                    acc_159[21] = z_296.y;
                    float2 ss2_297 = fma_f32x2_rn_noftz(z_296, z_296, make_float2(ssx, ssy));
                    ssx = ss2_297.x;
                    ssy = ss2_297.y;
                    float2 a2_298 = make_float2(acc_159[22], acc_159[23]);
                    float2 z_299 = add_f32x2_noftz(a2_298, make_float2(__uint_as_float(r_264[11] << 16), __uint_as_float(r_264[11] & 4294901760u)));
                    acc_159[22] = z_299.x;
                    acc_159[23] = z_299.y;
                    float2 ss2_300 = fma_f32x2_rn_noftz(z_299, z_299, make_float2(ssx, ssy));
                    ssx = ss2_300.x;
                    ssy = ss2_300.y;
                    float2 a2_301 = make_float2(acc_159[24], acc_159[25]);
                    float2 z_302 = add_f32x2_noftz(a2_301, make_float2(__uint_as_float(r_264[12] << 16), __uint_as_float(r_264[12] & 4294901760u)));
                    acc_159[24] = z_302.x;
                    acc_159[25] = z_302.y;
                    float2 ss2_303 = fma_f32x2_rn_noftz(z_302, z_302, make_float2(ssx, ssy));
                    ssx = ss2_303.x;
                    ssy = ss2_303.y;
                    float2 a2_304 = make_float2(acc_159[26], acc_159[27]);
                    float2 z_305 = add_f32x2_noftz(a2_304, make_float2(__uint_as_float(r_264[13] << 16), __uint_as_float(r_264[13] & 4294901760u)));
                    acc_159[26] = z_305.x;
                    acc_159[27] = z_305.y;
                    float2 ss2_306 = fma_f32x2_rn_noftz(z_305, z_305, make_float2(ssx, ssy));
                    ssx = ss2_306.x;
                    ssy = ss2_306.y;
                    float2 a2_307 = make_float2(acc_159[28], acc_159[29]);
                    float2 z_308 = add_f32x2_noftz(a2_307, make_float2(__uint_as_float(r_264[14] << 16), __uint_as_float(r_264[14] & 4294901760u)));
                    acc_159[28] = z_308.x;
                    acc_159[29] = z_308.y;
                    float2 ss2_309 = fma_f32x2_rn_noftz(z_308, z_308, make_float2(ssx, ssy));
                    ssx = ss2_309.x;
                    ssy = ss2_309.y;
                    float2 a2_310 = make_float2(acc_159[30], acc_159[31]);
                    float2 z_311 = add_f32x2_noftz(a2_310, make_float2(__uint_as_float(r_264[15] << 16), __uint_as_float(r_264[15] & 4294901760u)));
                    acc_159[30] = z_311.x;
                    acc_159[31] = z_311.y;
                    float2 ss2_312 = fma_f32x2_rn_noftz(z_311, z_311, make_float2(ssx, ssy));
                    ssx = ss2_312.x;
                    ssy = ss2_312.y;
                    unsigned int acc_159_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_159[_lp*2 + 0], acc_159[_lp*2+1 + 0]));
                        acc_159_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_313 = (f & 1) != 0;
                    bool fhi_314 = (f & 2) != 0;
                    unsigned int s_315[16];
                    s_315[0] = acc_159_bf16[0];
                    s_315[1] = acc_159_bf16[1];
                    s_315[2] = acc_159_bf16[2];
                    s_315[3] = acc_159_bf16[3];
                    s_315[4] = acc_159_bf16[4];
                    s_315[5] = acc_159_bf16[5];
                    s_315[6] = acc_159_bf16[6];
                    s_315[7] = acc_159_bf16[7];
                    s_315[8] = acc_159_bf16[8];
                    s_315[9] = acc_159_bf16[9];
                    s_315[10] = acc_159_bf16[10];
                    s_315[11] = acc_159_bf16[11];
                    s_315[12] = acc_159_bf16[12];
                    s_315[13] = acc_159_bf16[13];
                    s_315[14] = acc_159_bf16[14];
                    s_315[15] = acc_159_bf16[15];
                    if (fodd_313) {
                        s_315[0] = acc_159_bf16[4];
                        s_315[4] = acc_159_bf16[0];
                        s_315[8] = acc_159_bf16[12];
                        s_315[12] = acc_159_bf16[8];
                        s_315[1] = acc_159_bf16[5];
                        s_315[5] = acc_159_bf16[1];
                        s_315[9] = acc_159_bf16[13];
                        s_315[13] = acc_159_bf16[9];
                        s_315[2] = acc_159_bf16[6];
                        s_315[6] = acc_159_bf16[2];
                        s_315[10] = acc_159_bf16[14];
                        s_315[14] = acc_159_bf16[10];
                        s_315[3] = acc_159_bf16[7];
                        s_315[7] = acc_159_bf16[3];
                        s_315[11] = acc_159_bf16[15];
                        s_315[15] = acc_159_bf16[11];
                    }
                    unsigned int r_316[16];
                    r_316[0] = s_315[0];
                    r_316[1] = s_315[1];
                    r_316[2] = s_315[2];
                    r_316[3] = s_315[3];
                    r_316[4] = s_315[4];
                    r_316[5] = s_315[5];
                    r_316[6] = s_315[6];
                    r_316[7] = s_315[7];
                    r_316[8] = s_315[8];
                    r_316[9] = s_315[9];
                    r_316[10] = s_315[10];
                    r_316[11] = s_315[11];
                    r_316[12] = s_315[12];
                    r_316[13] = s_315[13];
                    r_316[14] = s_315[14];
                    r_316[15] = s_315[15];
                    if (fhi_314) {
                        r_316[0] = s_315[8];
                        r_316[4] = s_315[12];
                        r_316[8] = s_315[0];
                        r_316[12] = s_315[4];
                        r_316[1] = s_315[9];
                        r_316[5] = s_315[13];
                        r_316[9] = s_315[1];
                        r_316[13] = s_315[5];
                        r_316[2] = s_315[10];
                        r_316[6] = s_315[14];
                        r_316[10] = s_315[2];
                        r_316[14] = s_315[6];
                        r_316[3] = s_315[11];
                        r_316[7] = s_315[15];
                        r_316[11] = s_315[3];
                        r_316[15] = s_315[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 0))), "r"(r_316[0]), "r"(r_316[1]), "r"(r_316[2]), "r"(r_316[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 1))), "r"(r_316[4]), "r"(r_316[5]), "r"(r_316[6]), "r"(r_316[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 2))), "r"(r_316[8]), "r"(r_316[9]), "r"(r_316[10]), "r"(r_316[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_152 + 16 * (f ^ 3))), "r"(r_316[12]), "r"(r_316[13]), "r"(r_316[14]), "r"(r_316[15]) : "memory");
                    unsigned int d_317 = 0;
                    d_317 = d_317 | pv[8] & ~((pv[8] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[9] & ~((pv[9] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[10] & ~((pv[10] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[11] & ~((pv[11] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | ((ps[2] & 255) + 1 | (ps[2] >> 8) + 1) & 256;
                    d_317 = d_317 | pv[12] & ~((pv[12] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[13] & ~((pv[13] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[14] & ~((pv[14] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | pv[15] & ~((pv[15] & 2004318071) + 2004318071) & 2290649224u;
                    d_317 = d_317 | ((ps[3] & 255) + 1 | (ps[3] >> 8) + 1) & 256;
                    if (d_317 != 0) {
                        int pp0_1 = lane_2 + 64;
                        int pp1_1 = pp0_1 + 32;
                        uint32_t _fp4_wire_values_1[8];
                        uint32_t _fp4_wire_scales_1[2];
                        uint16_t _wire_h16_6_0;
                        uint16_t _wire_h16_6_1;
                        uint32_t _wire_vpend_6 = 3u;
                        uint32_t _wire_spend_6 = 3u;
                        do {
                            if (_wire_vpend_6 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[0]), "=r"(_fp4_wire_values_1[1]), "=r"(_fp4_wire_values_1[2]), "=r"(_fp4_wire_values_1[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0_1)))) : "memory");
                            }
                            if (_wire_vpend_6 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_1[4]), "=r"(_fp4_wire_values_1[5]), "=r"(_fp4_wire_values_1[6]), "=r"(_fp4_wire_values_1[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1_1)))) : "memory");
                            }
                            if (_wire_spend_6 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_6_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0_1)))) : "memory");
                                _fp4_wire_scales_1[0] = (uint32_t)_wire_h16_6_0;
                            }
                            if (_wire_spend_6 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_6_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1_1)))) : "memory");
                                _fp4_wire_scales_1[1] = (uint32_t)_wire_h16_6_1;
                            }
                            _wire_vpend_6 = ((uint32_t)((((_fp4_wire_values_1[0] & ~((_fp4_wire_values_1[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[1] & ~((_fp4_wire_values_1[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[2] & ~((_fp4_wire_values_1[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[3] & ~((_fp4_wire_values_1[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_1[4] & ~((_fp4_wire_values_1[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[5] & ~((_fp4_wire_values_1[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[6] & ~((_fp4_wire_values_1[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_1[7] & ~((_fp4_wire_values_1[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_6 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_1[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_1[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_6 | _wire_spend_6) != 0u);
                        pv[8] = _fp4_wire_values_1[0];
                        pv[9] = _fp4_wire_values_1[1];
                        pv[10] = _fp4_wire_values_1[2];
                        pv[11] = _fp4_wire_values_1[3];
                        ps[2] = _fp4_wire_scales_1[0];
                        pv[12] = _fp4_wire_values_1[4];
                        pv[13] = _fp4_wire_values_1[5];
                        pv[14] = _fp4_wire_values_1[6];
                        pv[15] = _fp4_wire_values_1[7];
                        ps[3] = _fp4_wire_scales_1[1];
                    }
                    int pp_318 = lane_2 + 64;
                    unsigned int own_319[4];
                    unsigned int own_sc_320[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_319[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_319[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_319[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_319[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_318)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_320[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_318)));
                    unsigned int a_321[4];
                    unsigned int b_322[4];
                    a_321[0] = ((first) ? own_319[0] : pv[8]);
                    b_322[0] = ((first) ? pv[8] : own_319[0]);
                    a_321[1] = ((first) ? own_319[1] : pv[9]);
                    b_322[1] = ((first) ? pv[9] : own_319[1]);
                    a_321[2] = ((first) ? own_319[2] : pv[10]);
                    b_322[2] = ((first) ? pv[10] : own_319[2]);
                    a_321[3] = ((first) ? own_319[3] : pv[11]);
                    b_322[3] = ((first) ? pv[11] : own_319[3]);
                    unsigned int a_sc_323 = ((first) ? own_sc_320[0] : ps[2]);
                    unsigned int b_sc_324 = ((first) ? ps[2] : own_sc_320[0]);
                    float acc_325[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(a_sc_323)));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_8), "f"(rs_a_f));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_9), "f"(rs_a_f));
                    float2 scale0_326 = make_float2(_mul_8, _mul_8);
                    float2 scale1_327 = make_float2(_mul_9, _mul_9);
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_321[0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(a_321[0]));
                        #endif
                    }
                    float2 a_328 = make_float2(acc_325[0], acc_325[1]);
                    float2 q_329 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 r_330 = fma_f32x2_rn_noftz(q_329, scale0_326, a_328);
                    acc_325[0] = r_330.x;
                    acc_325[1] = r_330.y;
                    float2 a_331 = make_float2(acc_325[2], acc_325[3]);
                    float2 q_332 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 r_333 = fma_f32x2_rn_noftz(q_332, scale0_326, a_331);
                    acc_325[2] = r_333.x;
                    acc_325[3] = r_333.y;
                    float2 a_334 = make_float2(acc_325[4], acc_325[5]);
                    float2 q_335 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 r_336 = fma_f32x2_rn_noftz(q_335, scale0_326, a_334);
                    acc_325[4] = r_336.x;
                    acc_325[5] = r_336.y;
                    float2 a_337 = make_float2(acc_325[6], acc_325[7]);
                    float2 q_338 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 r_339 = fma_f32x2_rn_noftz(q_338, scale0_326, a_337);
                    acc_325[6] = r_339.x;
                    acc_325[7] = r_339.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_321[1 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(a_321[1 + 0]));
                        #endif
                    }
                    float2 a_340 = make_float2(acc_325[8], acc_325[9]);
                    float2 q_341 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 r_342 = fma_f32x2_rn_noftz(q_341, scale0_326, a_340);
                    acc_325[8] = r_342.x;
                    acc_325[9] = r_342.y;
                    float2 a_343 = make_float2(acc_325[10], acc_325[11]);
                    float2 q_344 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 r_345 = fma_f32x2_rn_noftz(q_344, scale0_326, a_343);
                    acc_325[10] = r_345.x;
                    acc_325[11] = r_345.y;
                    float2 a_346 = make_float2(acc_325[12], acc_325[13]);
                    float2 q_347 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 r_348 = fma_f32x2_rn_noftz(q_347, scale0_326, a_346);
                    acc_325[12] = r_348.x;
                    acc_325[13] = r_348.y;
                    float2 a_349 = make_float2(acc_325[14], acc_325[15]);
                    float2 q_350 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 r_351 = fma_f32x2_rn_noftz(q_350, scale0_326, a_349);
                    acc_325[14] = r_351.x;
                    acc_325[15] = r_351.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_321[2 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(a_321[2 + 0]));
                        #endif
                    }
                    float2 a_352 = make_float2(acc_325[16], acc_325[17]);
                    float2 q_353 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 r_354 = fma_f32x2_rn_noftz(q_353, scale1_327, a_352);
                    acc_325[16] = r_354.x;
                    acc_325[17] = r_354.y;
                    float2 a_355 = make_float2(acc_325[18], acc_325[19]);
                    float2 q_356 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 r_357 = fma_f32x2_rn_noftz(q_356, scale1_327, a_355);
                    acc_325[18] = r_357.x;
                    acc_325[19] = r_357.y;
                    float2 a_358 = make_float2(acc_325[20], acc_325[21]);
                    float2 q_359 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 r_360 = fma_f32x2_rn_noftz(q_359, scale1_327, a_358);
                    acc_325[20] = r_360.x;
                    acc_325[21] = r_360.y;
                    float2 a_361 = make_float2(acc_325[22], acc_325[23]);
                    float2 q_362 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 r_363 = fma_f32x2_rn_noftz(q_362, scale1_327, a_361);
                    acc_325[22] = r_363.x;
                    acc_325[23] = r_363.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_321[3 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(a_321[3 + 0]));
                        #endif
                    }
                    float2 a_364 = make_float2(acc_325[24], acc_325[25]);
                    float2 q_365 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 r_366 = fma_f32x2_rn_noftz(q_365, scale1_327, a_364);
                    acc_325[24] = r_366.x;
                    acc_325[25] = r_366.y;
                    float2 a_367 = make_float2(acc_325[26], acc_325[27]);
                    float2 q_368 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 r_369 = fma_f32x2_rn_noftz(q_368, scale1_327, a_367);
                    acc_325[26] = r_369.x;
                    acc_325[27] = r_369.y;
                    float2 a_370 = make_float2(acc_325[28], acc_325[29]);
                    float2 q_371 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 r_372 = fma_f32x2_rn_noftz(q_371, scale1_327, a_370);
                    acc_325[28] = r_372.x;
                    acc_325[29] = r_372.y;
                    float2 a_373 = make_float2(acc_325[30], acc_325[31]);
                    float2 q_374 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 r_375 = fma_f32x2_rn_noftz(q_374, scale1_327, a_373);
                    acc_325[30] = r_375.x;
                    acc_325[31] = r_375.y;
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(b_sc_324)));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_10), "f"(rs_b_f));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_11), "f"(rs_b_f));
                    float2 scale0_376 = make_float2(_mul_10, _mul_10);
                    float2 scale1_377 = make_float2(_mul_11, _mul_11);
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_322[0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(b_322[0]));
                        #endif
                    }
                    float2 a_378 = make_float2(acc_325[0], acc_325[1]);
                    float2 q_379 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 r_380 = fma_f32x2_rn_noftz(q_379, scale0_376, a_378);
                    acc_325[0] = r_380.x;
                    acc_325[1] = r_380.y;
                    float2 a_381 = make_float2(acc_325[2], acc_325[3]);
                    float2 q_382 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 r_383 = fma_f32x2_rn_noftz(q_382, scale0_376, a_381);
                    acc_325[2] = r_383.x;
                    acc_325[3] = r_383.y;
                    float2 a_384 = make_float2(acc_325[4], acc_325[5]);
                    float2 q_385 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 r_386 = fma_f32x2_rn_noftz(q_385, scale0_376, a_384);
                    acc_325[4] = r_386.x;
                    acc_325[5] = r_386.y;
                    float2 a_387 = make_float2(acc_325[6], acc_325[7]);
                    float2 q_388 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 r_389 = fma_f32x2_rn_noftz(q_388, scale0_376, a_387);
                    acc_325[6] = r_389.x;
                    acc_325[7] = r_389.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_322[1 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(b_322[1 + 0]));
                        #endif
                    }
                    float2 a_390 = make_float2(acc_325[8], acc_325[9]);
                    float2 q_391 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 r_392 = fma_f32x2_rn_noftz(q_391, scale0_376, a_390);
                    acc_325[8] = r_392.x;
                    acc_325[9] = r_392.y;
                    float2 a_393 = make_float2(acc_325[10], acc_325[11]);
                    float2 q_394 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 r_395 = fma_f32x2_rn_noftz(q_394, scale0_376, a_393);
                    acc_325[10] = r_395.x;
                    acc_325[11] = r_395.y;
                    float2 a_396 = make_float2(acc_325[12], acc_325[13]);
                    float2 q_397 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 r_398 = fma_f32x2_rn_noftz(q_397, scale0_376, a_396);
                    acc_325[12] = r_398.x;
                    acc_325[13] = r_398.y;
                    float2 a_399 = make_float2(acc_325[14], acc_325[15]);
                    float2 q_400 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 r_401 = fma_f32x2_rn_noftz(q_400, scale0_376, a_399);
                    acc_325[14] = r_401.x;
                    acc_325[15] = r_401.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_322[2 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(b_322[2 + 0]));
                        #endif
                    }
                    float2 a_402 = make_float2(acc_325[16], acc_325[17]);
                    float2 q_403 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 r_404 = fma_f32x2_rn_noftz(q_403, scale1_377, a_402);
                    acc_325[16] = r_404.x;
                    acc_325[17] = r_404.y;
                    float2 a_405 = make_float2(acc_325[18], acc_325[19]);
                    float2 q_406 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 r_407 = fma_f32x2_rn_noftz(q_406, scale1_377, a_405);
                    acc_325[18] = r_407.x;
                    acc_325[19] = r_407.y;
                    float2 a_408 = make_float2(acc_325[20], acc_325[21]);
                    float2 q_409 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 r_410 = fma_f32x2_rn_noftz(q_409, scale1_377, a_408);
                    acc_325[20] = r_410.x;
                    acc_325[21] = r_410.y;
                    float2 a_411 = make_float2(acc_325[22], acc_325[23]);
                    float2 q_412 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 r_413 = fma_f32x2_rn_noftz(q_412, scale1_377, a_411);
                    acc_325[22] = r_413.x;
                    acc_325[23] = r_413.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_322[3 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(b_322[3 + 0]));
                        #endif
                    }
                    float2 a_414 = make_float2(acc_325[24], acc_325[25]);
                    float2 q_415 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 r_416 = fma_f32x2_rn_noftz(q_415, scale1_377, a_414);
                    acc_325[24] = r_416.x;
                    acc_325[25] = r_416.y;
                    float2 a_417 = make_float2(acc_325[26], acc_325[27]);
                    float2 q_418 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 r_419 = fma_f32x2_rn_noftz(q_418, scale1_377, a_417);
                    acc_325[26] = r_419.x;
                    acc_325[27] = r_419.y;
                    float2 a_420 = make_float2(acc_325[28], acc_325[29]);
                    float2 q_421 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 r_422 = fma_f32x2_rn_noftz(q_421, scale1_377, a_420);
                    acc_325[28] = r_422.x;
                    acc_325[29] = r_422.y;
                    float2 a_423 = make_float2(acc_325[30], acc_325[31]);
                    float2 q_424 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 r_425 = fma_f32x2_rn_noftz(q_424, scale1_377, a_423);
                    acc_325[30] = r_425.x;
                    acc_325[31] = r_425.y;
                    unsigned int t_426[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_426[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_426[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_426[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_426[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_426[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 3))));
                    bool fodd_427 = (f & 1) != 0;
                    bool fhi_428 = (f & 2) != 0;
                    unsigned int s_429[16];
                    s_429[0] = t_426[0];
                    s_429[1] = t_426[1];
                    s_429[2] = t_426[2];
                    s_429[3] = t_426[3];
                    s_429[4] = t_426[4];
                    s_429[5] = t_426[5];
                    s_429[6] = t_426[6];
                    s_429[7] = t_426[7];
                    s_429[8] = t_426[8];
                    s_429[9] = t_426[9];
                    s_429[10] = t_426[10];
                    s_429[11] = t_426[11];
                    s_429[12] = t_426[12];
                    s_429[13] = t_426[13];
                    s_429[14] = t_426[14];
                    s_429[15] = t_426[15];
                    if (fodd_427) {
                        s_429[0] = t_426[4];
                        s_429[4] = t_426[0];
                        s_429[8] = t_426[12];
                        s_429[12] = t_426[8];
                        s_429[1] = t_426[5];
                        s_429[5] = t_426[1];
                        s_429[9] = t_426[13];
                        s_429[13] = t_426[9];
                        s_429[2] = t_426[6];
                        s_429[6] = t_426[2];
                        s_429[10] = t_426[14];
                        s_429[14] = t_426[10];
                        s_429[3] = t_426[7];
                        s_429[7] = t_426[3];
                        s_429[11] = t_426[15];
                        s_429[15] = t_426[11];
                    }
                    unsigned int r_430[16];
                    r_430[0] = s_429[0];
                    r_430[1] = s_429[1];
                    r_430[2] = s_429[2];
                    r_430[3] = s_429[3];
                    r_430[4] = s_429[4];
                    r_430[5] = s_429[5];
                    r_430[6] = s_429[6];
                    r_430[7] = s_429[7];
                    r_430[8] = s_429[8];
                    r_430[9] = s_429[9];
                    r_430[10] = s_429[10];
                    r_430[11] = s_429[11];
                    r_430[12] = s_429[12];
                    r_430[13] = s_429[13];
                    r_430[14] = s_429[14];
                    r_430[15] = s_429[15];
                    if (fhi_428) {
                        r_430[0] = s_429[8];
                        r_430[4] = s_429[12];
                        r_430[8] = s_429[0];
                        r_430[12] = s_429[4];
                        r_430[1] = s_429[9];
                        r_430[5] = s_429[13];
                        r_430[9] = s_429[1];
                        r_430[13] = s_429[5];
                        r_430[2] = s_429[10];
                        r_430[6] = s_429[14];
                        r_430[10] = s_429[2];
                        r_430[14] = s_429[6];
                        r_430[3] = s_429[11];
                        r_430[7] = s_429[15];
                        r_430[11] = s_429[3];
                        r_430[15] = s_429[7];
                    }
                    float2 a2_431 = make_float2(acc_325[0], acc_325[1]);
                    float2 z_432 = add_f32x2_noftz(a2_431, make_float2(__uint_as_float(r_430[0] << 16), __uint_as_float(r_430[0] & 4294901760u)));
                    acc_325[0] = z_432.x;
                    acc_325[1] = z_432.y;
                    float2 ss2_433 = fma_f32x2_rn_noftz(z_432, z_432, make_float2(ssx, ssy));
                    ssx = ss2_433.x;
                    ssy = ss2_433.y;
                    float2 a2_434 = make_float2(acc_325[2], acc_325[3]);
                    float2 z_435 = add_f32x2_noftz(a2_434, make_float2(__uint_as_float(r_430[1] << 16), __uint_as_float(r_430[1] & 4294901760u)));
                    acc_325[2] = z_435.x;
                    acc_325[3] = z_435.y;
                    float2 ss2_436 = fma_f32x2_rn_noftz(z_435, z_435, make_float2(ssx, ssy));
                    ssx = ss2_436.x;
                    ssy = ss2_436.y;
                    float2 a2_437 = make_float2(acc_325[4], acc_325[5]);
                    float2 z_438 = add_f32x2_noftz(a2_437, make_float2(__uint_as_float(r_430[2] << 16), __uint_as_float(r_430[2] & 4294901760u)));
                    acc_325[4] = z_438.x;
                    acc_325[5] = z_438.y;
                    float2 ss2_439 = fma_f32x2_rn_noftz(z_438, z_438, make_float2(ssx, ssy));
                    ssx = ss2_439.x;
                    ssy = ss2_439.y;
                    float2 a2_440 = make_float2(acc_325[6], acc_325[7]);
                    float2 z_441 = add_f32x2_noftz(a2_440, make_float2(__uint_as_float(r_430[3] << 16), __uint_as_float(r_430[3] & 4294901760u)));
                    acc_325[6] = z_441.x;
                    acc_325[7] = z_441.y;
                    float2 ss2_442 = fma_f32x2_rn_noftz(z_441, z_441, make_float2(ssx, ssy));
                    ssx = ss2_442.x;
                    ssy = ss2_442.y;
                    float2 a2_443 = make_float2(acc_325[8], acc_325[9]);
                    float2 z_444 = add_f32x2_noftz(a2_443, make_float2(__uint_as_float(r_430[4] << 16), __uint_as_float(r_430[4] & 4294901760u)));
                    acc_325[8] = z_444.x;
                    acc_325[9] = z_444.y;
                    float2 ss2_445 = fma_f32x2_rn_noftz(z_444, z_444, make_float2(ssx, ssy));
                    ssx = ss2_445.x;
                    ssy = ss2_445.y;
                    float2 a2_446 = make_float2(acc_325[10], acc_325[11]);
                    float2 z_447 = add_f32x2_noftz(a2_446, make_float2(__uint_as_float(r_430[5] << 16), __uint_as_float(r_430[5] & 4294901760u)));
                    acc_325[10] = z_447.x;
                    acc_325[11] = z_447.y;
                    float2 ss2_448 = fma_f32x2_rn_noftz(z_447, z_447, make_float2(ssx, ssy));
                    ssx = ss2_448.x;
                    ssy = ss2_448.y;
                    float2 a2_449 = make_float2(acc_325[12], acc_325[13]);
                    float2 z_450 = add_f32x2_noftz(a2_449, make_float2(__uint_as_float(r_430[6] << 16), __uint_as_float(r_430[6] & 4294901760u)));
                    acc_325[12] = z_450.x;
                    acc_325[13] = z_450.y;
                    float2 ss2_451 = fma_f32x2_rn_noftz(z_450, z_450, make_float2(ssx, ssy));
                    ssx = ss2_451.x;
                    ssy = ss2_451.y;
                    float2 a2_452 = make_float2(acc_325[14], acc_325[15]);
                    float2 z_453 = add_f32x2_noftz(a2_452, make_float2(__uint_as_float(r_430[7] << 16), __uint_as_float(r_430[7] & 4294901760u)));
                    acc_325[14] = z_453.x;
                    acc_325[15] = z_453.y;
                    float2 ss2_454 = fma_f32x2_rn_noftz(z_453, z_453, make_float2(ssx, ssy));
                    ssx = ss2_454.x;
                    ssy = ss2_454.y;
                    float2 a2_455 = make_float2(acc_325[16], acc_325[17]);
                    float2 z_456 = add_f32x2_noftz(a2_455, make_float2(__uint_as_float(r_430[8] << 16), __uint_as_float(r_430[8] & 4294901760u)));
                    acc_325[16] = z_456.x;
                    acc_325[17] = z_456.y;
                    float2 ss2_457 = fma_f32x2_rn_noftz(z_456, z_456, make_float2(ssx, ssy));
                    ssx = ss2_457.x;
                    ssy = ss2_457.y;
                    float2 a2_458 = make_float2(acc_325[18], acc_325[19]);
                    float2 z_459 = add_f32x2_noftz(a2_458, make_float2(__uint_as_float(r_430[9] << 16), __uint_as_float(r_430[9] & 4294901760u)));
                    acc_325[18] = z_459.x;
                    acc_325[19] = z_459.y;
                    float2 ss2_460 = fma_f32x2_rn_noftz(z_459, z_459, make_float2(ssx, ssy));
                    ssx = ss2_460.x;
                    ssy = ss2_460.y;
                    float2 a2_461 = make_float2(acc_325[20], acc_325[21]);
                    float2 z_462 = add_f32x2_noftz(a2_461, make_float2(__uint_as_float(r_430[10] << 16), __uint_as_float(r_430[10] & 4294901760u)));
                    acc_325[20] = z_462.x;
                    acc_325[21] = z_462.y;
                    float2 ss2_463 = fma_f32x2_rn_noftz(z_462, z_462, make_float2(ssx, ssy));
                    ssx = ss2_463.x;
                    ssy = ss2_463.y;
                    float2 a2_464 = make_float2(acc_325[22], acc_325[23]);
                    float2 z_465 = add_f32x2_noftz(a2_464, make_float2(__uint_as_float(r_430[11] << 16), __uint_as_float(r_430[11] & 4294901760u)));
                    acc_325[22] = z_465.x;
                    acc_325[23] = z_465.y;
                    float2 ss2_466 = fma_f32x2_rn_noftz(z_465, z_465, make_float2(ssx, ssy));
                    ssx = ss2_466.x;
                    ssy = ss2_466.y;
                    float2 a2_467 = make_float2(acc_325[24], acc_325[25]);
                    float2 z_468 = add_f32x2_noftz(a2_467, make_float2(__uint_as_float(r_430[12] << 16), __uint_as_float(r_430[12] & 4294901760u)));
                    acc_325[24] = z_468.x;
                    acc_325[25] = z_468.y;
                    float2 ss2_469 = fma_f32x2_rn_noftz(z_468, z_468, make_float2(ssx, ssy));
                    ssx = ss2_469.x;
                    ssy = ss2_469.y;
                    float2 a2_470 = make_float2(acc_325[26], acc_325[27]);
                    float2 z_471 = add_f32x2_noftz(a2_470, make_float2(__uint_as_float(r_430[13] << 16), __uint_as_float(r_430[13] & 4294901760u)));
                    acc_325[26] = z_471.x;
                    acc_325[27] = z_471.y;
                    float2 ss2_472 = fma_f32x2_rn_noftz(z_471, z_471, make_float2(ssx, ssy));
                    ssx = ss2_472.x;
                    ssy = ss2_472.y;
                    float2 a2_473 = make_float2(acc_325[28], acc_325[29]);
                    float2 z_474 = add_f32x2_noftz(a2_473, make_float2(__uint_as_float(r_430[14] << 16), __uint_as_float(r_430[14] & 4294901760u)));
                    acc_325[28] = z_474.x;
                    acc_325[29] = z_474.y;
                    float2 ss2_475 = fma_f32x2_rn_noftz(z_474, z_474, make_float2(ssx, ssy));
                    ssx = ss2_475.x;
                    ssy = ss2_475.y;
                    float2 a2_476 = make_float2(acc_325[30], acc_325[31]);
                    float2 z_477 = add_f32x2_noftz(a2_476, make_float2(__uint_as_float(r_430[15] << 16), __uint_as_float(r_430[15] & 4294901760u)));
                    acc_325[30] = z_477.x;
                    acc_325[31] = z_477.y;
                    float2 ss2_478 = fma_f32x2_rn_noftz(z_477, z_477, make_float2(ssx, ssy));
                    ssx = ss2_478.x;
                    ssy = ss2_478.y;
                    unsigned int acc_325_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_325[_lp*2 + 0], acc_325[_lp*2+1 + 0]));
                        acc_325_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_479 = (f & 1) != 0;
                    bool fhi_480 = (f & 2) != 0;
                    unsigned int s_481[16];
                    s_481[0] = acc_325_bf16[0];
                    s_481[1] = acc_325_bf16[1];
                    s_481[2] = acc_325_bf16[2];
                    s_481[3] = acc_325_bf16[3];
                    s_481[4] = acc_325_bf16[4];
                    s_481[5] = acc_325_bf16[5];
                    s_481[6] = acc_325_bf16[6];
                    s_481[7] = acc_325_bf16[7];
                    s_481[8] = acc_325_bf16[8];
                    s_481[9] = acc_325_bf16[9];
                    s_481[10] = acc_325_bf16[10];
                    s_481[11] = acc_325_bf16[11];
                    s_481[12] = acc_325_bf16[12];
                    s_481[13] = acc_325_bf16[13];
                    s_481[14] = acc_325_bf16[14];
                    s_481[15] = acc_325_bf16[15];
                    if (fodd_479) {
                        s_481[0] = acc_325_bf16[4];
                        s_481[4] = acc_325_bf16[0];
                        s_481[8] = acc_325_bf16[12];
                        s_481[12] = acc_325_bf16[8];
                        s_481[1] = acc_325_bf16[5];
                        s_481[5] = acc_325_bf16[1];
                        s_481[9] = acc_325_bf16[13];
                        s_481[13] = acc_325_bf16[9];
                        s_481[2] = acc_325_bf16[6];
                        s_481[6] = acc_325_bf16[2];
                        s_481[10] = acc_325_bf16[14];
                        s_481[14] = acc_325_bf16[10];
                        s_481[3] = acc_325_bf16[7];
                        s_481[7] = acc_325_bf16[3];
                        s_481[11] = acc_325_bf16[15];
                        s_481[15] = acc_325_bf16[11];
                    }
                    unsigned int r_482[16];
                    r_482[0] = s_481[0];
                    r_482[1] = s_481[1];
                    r_482[2] = s_481[2];
                    r_482[3] = s_481[3];
                    r_482[4] = s_481[4];
                    r_482[5] = s_481[5];
                    r_482[6] = s_481[6];
                    r_482[7] = s_481[7];
                    r_482[8] = s_481[8];
                    r_482[9] = s_481[9];
                    r_482[10] = s_481[10];
                    r_482[11] = s_481[11];
                    r_482[12] = s_481[12];
                    r_482[13] = s_481[13];
                    r_482[14] = s_481[14];
                    r_482[15] = s_481[15];
                    if (fhi_480) {
                        r_482[0] = s_481[8];
                        r_482[4] = s_481[12];
                        r_482[8] = s_481[0];
                        r_482[12] = s_481[4];
                        r_482[1] = s_481[9];
                        r_482[5] = s_481[13];
                        r_482[9] = s_481[1];
                        r_482[13] = s_481[5];
                        r_482[2] = s_481[10];
                        r_482[6] = s_481[14];
                        r_482[10] = s_481[2];
                        r_482[14] = s_481[6];
                        r_482[3] = s_481[11];
                        r_482[7] = s_481[15];
                        r_482[11] = s_481[3];
                        r_482[15] = s_481[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 0))), "r"(r_482[0]), "r"(r_482[1]), "r"(r_482[2]), "r"(r_482[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 1))), "r"(r_482[4]), "r"(r_482[5]), "r"(r_482[6]), "r"(r_482[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 2))), "r"(r_482[8]), "r"(r_482[9]), "r"(r_482[10]), "r"(r_482[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_318 + 16 * (f ^ 3))), "r"(r_482[12]), "r"(r_482[13]), "r"(r_482[14]), "r"(r_482[15]) : "memory");
                    int pp_483 = lane_2 + 96;
                    unsigned int own_484[4];
                    unsigned int own_sc_485[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_484[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_484[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_484[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_484[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_483)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_485[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_483)));
                    unsigned int a_486[4];
                    unsigned int b_487[4];
                    a_486[0] = ((first) ? own_484[0] : pv[12]);
                    b_487[0] = ((first) ? pv[12] : own_484[0]);
                    a_486[1] = ((first) ? own_484[1] : pv[13]);
                    b_487[1] = ((first) ? pv[13] : own_484[1]);
                    a_486[2] = ((first) ? own_484[2] : pv[14]);
                    b_487[2] = ((first) ? pv[14] : own_484[2]);
                    a_486[3] = ((first) ? own_484[3] : pv[15]);
                    b_487[3] = ((first) ? pv[15] : own_484[3]);
                    unsigned int a_sc_488 = ((first) ? own_sc_485[0] : ps[3]);
                    unsigned int b_sc_489 = ((first) ? ps[3] : own_sc_485[0]);
                    float acc_490[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(a_sc_488)));
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                    float _cvt_f32_f16_13;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_cvt_f32_f16_12), "f"(rs_a_f));
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_cvt_f32_f16_13), "f"(rs_a_f));
                    float2 scale0_491 = make_float2(_mul_12, _mul_12);
                    float2 scale1_492 = make_float2(_mul_13, _mul_13);
                    float _unpack_fp4_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_486[0]));
                        (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(a_486[0]));
                        #endif
                    }
                    float2 a_493 = make_float2(acc_490[0], acc_490[1]);
                    float2 q_494 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                    float2 r_495 = fma_f32x2_rn_noftz(q_494, scale0_491, a_493);
                    acc_490[0] = r_495.x;
                    acc_490[1] = r_495.y;
                    float2 a_496 = make_float2(acc_490[2], acc_490[3]);
                    float2 q_497 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                    float2 r_498 = fma_f32x2_rn_noftz(q_497, scale0_491, a_496);
                    acc_490[2] = r_498.x;
                    acc_490[3] = r_498.y;
                    float2 a_499 = make_float2(acc_490[4], acc_490[5]);
                    float2 q_500 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                    float2 r_501 = fma_f32x2_rn_noftz(q_500, scale0_491, a_499);
                    acc_490[4] = r_501.x;
                    acc_490[5] = r_501.y;
                    float2 a_502 = make_float2(acc_490[6], acc_490[7]);
                    float2 q_503 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                    float2 r_504 = fma_f32x2_rn_noftz(q_503, scale0_491, a_502);
                    acc_490[6] = r_504.x;
                    acc_490[7] = r_504.y;
                    float _unpack_fp4_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_486[1 + 0]));
                        (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(a_486[1 + 0]));
                        #endif
                    }
                    float2 a_505 = make_float2(acc_490[8], acc_490[9]);
                    float2 q_506 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                    float2 r_507 = fma_f32x2_rn_noftz(q_506, scale0_491, a_505);
                    acc_490[8] = r_507.x;
                    acc_490[9] = r_507.y;
                    float2 a_508 = make_float2(acc_490[10], acc_490[11]);
                    float2 q_509 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                    float2 r_510 = fma_f32x2_rn_noftz(q_509, scale0_491, a_508);
                    acc_490[10] = r_510.x;
                    acc_490[11] = r_510.y;
                    float2 a_511 = make_float2(acc_490[12], acc_490[13]);
                    float2 q_512 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                    float2 r_513 = fma_f32x2_rn_noftz(q_512, scale0_491, a_511);
                    acc_490[12] = r_513.x;
                    acc_490[13] = r_513.y;
                    float2 a_514 = make_float2(acc_490[14], acc_490[15]);
                    float2 q_515 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                    float2 r_516 = fma_f32x2_rn_noftz(q_515, scale0_491, a_514);
                    acc_490[14] = r_516.x;
                    acc_490[15] = r_516.y;
                    float _unpack_fp4_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_486[2 + 0]));
                        (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(a_486[2 + 0]));
                        #endif
                    }
                    float2 a_517 = make_float2(acc_490[16], acc_490[17]);
                    float2 q_518 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                    float2 r_519 = fma_f32x2_rn_noftz(q_518, scale1_492, a_517);
                    acc_490[16] = r_519.x;
                    acc_490[17] = r_519.y;
                    float2 a_520 = make_float2(acc_490[18], acc_490[19]);
                    float2 q_521 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                    float2 r_522 = fma_f32x2_rn_noftz(q_521, scale1_492, a_520);
                    acc_490[18] = r_522.x;
                    acc_490[19] = r_522.y;
                    float2 a_523 = make_float2(acc_490[20], acc_490[21]);
                    float2 q_524 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                    float2 r_525 = fma_f32x2_rn_noftz(q_524, scale1_492, a_523);
                    acc_490[20] = r_525.x;
                    acc_490[21] = r_525.y;
                    float2 a_526 = make_float2(acc_490[22], acc_490[23]);
                    float2 q_527 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                    float2 r_528 = fma_f32x2_rn_noftz(q_527, scale1_492, a_526);
                    acc_490[22] = r_528.x;
                    acc_490[23] = r_528.y;
                    float _unpack_fp4_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_486[3 + 0]));
                        (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(a_486[3 + 0]));
                        #endif
                    }
                    float2 a_529 = make_float2(acc_490[24], acc_490[25]);
                    float2 q_530 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                    float2 r_531 = fma_f32x2_rn_noftz(q_530, scale1_492, a_529);
                    acc_490[24] = r_531.x;
                    acc_490[25] = r_531.y;
                    float2 a_532 = make_float2(acc_490[26], acc_490[27]);
                    float2 q_533 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                    float2 r_534 = fma_f32x2_rn_noftz(q_533, scale1_492, a_532);
                    acc_490[26] = r_534.x;
                    acc_490[27] = r_534.y;
                    float2 a_535 = make_float2(acc_490[28], acc_490[29]);
                    float2 q_536 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                    float2 r_537 = fma_f32x2_rn_noftz(q_536, scale1_492, a_535);
                    acc_490[28] = r_537.x;
                    acc_490[29] = r_537.y;
                    float2 a_538 = make_float2(acc_490[30], acc_490[31]);
                    float2 q_539 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                    float2 r_540 = fma_f32x2_rn_noftz(q_539, scale1_492, a_538);
                    acc_490[30] = r_540.x;
                    acc_490[31] = r_540.y;
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(b_sc_489)));
                    float _cvt_f32_f16_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                    float _cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_cvt_f32_f16_14), "f"(rs_b_f));
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_cvt_f32_f16_15), "f"(rs_b_f));
                    float2 scale0_541 = make_float2(_mul_14, _mul_14);
                    float2 scale1_542 = make_float2(_mul_15, _mul_15);
                    float _unpack_fp4_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_487[0]));
                        (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(b_487[0]));
                        #endif
                    }
                    float2 a_543 = make_float2(acc_490[0], acc_490[1]);
                    float2 q_544 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                    float2 r_545 = fma_f32x2_rn_noftz(q_544, scale0_541, a_543);
                    acc_490[0] = r_545.x;
                    acc_490[1] = r_545.y;
                    float2 a_546 = make_float2(acc_490[2], acc_490[3]);
                    float2 q_547 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                    float2 r_548 = fma_f32x2_rn_noftz(q_547, scale0_541, a_546);
                    acc_490[2] = r_548.x;
                    acc_490[3] = r_548.y;
                    float2 a_549 = make_float2(acc_490[4], acc_490[5]);
                    float2 q_550 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                    float2 r_551 = fma_f32x2_rn_noftz(q_550, scale0_541, a_549);
                    acc_490[4] = r_551.x;
                    acc_490[5] = r_551.y;
                    float2 a_552 = make_float2(acc_490[6], acc_490[7]);
                    float2 q_553 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                    float2 r_554 = fma_f32x2_rn_noftz(q_553, scale0_541, a_552);
                    acc_490[6] = r_554.x;
                    acc_490[7] = r_554.y;
                    float _unpack_fp4_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_487[1 + 0]));
                        (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(b_487[1 + 0]));
                        #endif
                    }
                    float2 a_555 = make_float2(acc_490[8], acc_490[9]);
                    float2 q_556 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                    float2 r_557 = fma_f32x2_rn_noftz(q_556, scale0_541, a_555);
                    acc_490[8] = r_557.x;
                    acc_490[9] = r_557.y;
                    float2 a_558 = make_float2(acc_490[10], acc_490[11]);
                    float2 q_559 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                    float2 r_560 = fma_f32x2_rn_noftz(q_559, scale0_541, a_558);
                    acc_490[10] = r_560.x;
                    acc_490[11] = r_560.y;
                    float2 a_561 = make_float2(acc_490[12], acc_490[13]);
                    float2 q_562 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                    float2 r_563 = fma_f32x2_rn_noftz(q_562, scale0_541, a_561);
                    acc_490[12] = r_563.x;
                    acc_490[13] = r_563.y;
                    float2 a_564 = make_float2(acc_490[14], acc_490[15]);
                    float2 q_565 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                    float2 r_566 = fma_f32x2_rn_noftz(q_565, scale0_541, a_564);
                    acc_490[14] = r_566.x;
                    acc_490[15] = r_566.y;
                    float _unpack_fp4_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_487[2 + 0]));
                        (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(b_487[2 + 0]));
                        #endif
                    }
                    float2 a_567 = make_float2(acc_490[16], acc_490[17]);
                    float2 q_568 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                    float2 r_569 = fma_f32x2_rn_noftz(q_568, scale1_542, a_567);
                    acc_490[16] = r_569.x;
                    acc_490[17] = r_569.y;
                    float2 a_570 = make_float2(acc_490[18], acc_490[19]);
                    float2 q_571 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                    float2 r_572 = fma_f32x2_rn_noftz(q_571, scale1_542, a_570);
                    acc_490[18] = r_572.x;
                    acc_490[19] = r_572.y;
                    float2 a_573 = make_float2(acc_490[20], acc_490[21]);
                    float2 q_574 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                    float2 r_575 = fma_f32x2_rn_noftz(q_574, scale1_542, a_573);
                    acc_490[20] = r_575.x;
                    acc_490[21] = r_575.y;
                    float2 a_576 = make_float2(acc_490[22], acc_490[23]);
                    float2 q_577 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                    float2 r_578 = fma_f32x2_rn_noftz(q_577, scale1_542, a_576);
                    acc_490[22] = r_578.x;
                    acc_490[23] = r_578.y;
                    float _unpack_fp4_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_487[3 + 0]));
                        (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(b_487[3 + 0]));
                        #endif
                    }
                    float2 a_579 = make_float2(acc_490[24], acc_490[25]);
                    float2 q_580 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                    float2 r_581 = fma_f32x2_rn_noftz(q_580, scale1_542, a_579);
                    acc_490[24] = r_581.x;
                    acc_490[25] = r_581.y;
                    float2 a_582 = make_float2(acc_490[26], acc_490[27]);
                    float2 q_583 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                    float2 r_584 = fma_f32x2_rn_noftz(q_583, scale1_542, a_582);
                    acc_490[26] = r_584.x;
                    acc_490[27] = r_584.y;
                    float2 a_585 = make_float2(acc_490[28], acc_490[29]);
                    float2 q_586 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                    float2 r_587 = fma_f32x2_rn_noftz(q_586, scale1_542, a_585);
                    acc_490[28] = r_587.x;
                    acc_490[29] = r_587.y;
                    float2 a_588 = make_float2(acc_490[30], acc_490[31]);
                    float2 q_589 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                    float2 r_590 = fma_f32x2_rn_noftz(q_589, scale1_542, a_588);
                    acc_490[30] = r_590.x;
                    acc_490[31] = r_590.y;
                    unsigned int t_591[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_591[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_591[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_591[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_591[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_591[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 3))));
                    bool fodd_592 = (f & 1) != 0;
                    bool fhi_593 = (f & 2) != 0;
                    unsigned int s_594[16];
                    s_594[0] = t_591[0];
                    s_594[1] = t_591[1];
                    s_594[2] = t_591[2];
                    s_594[3] = t_591[3];
                    s_594[4] = t_591[4];
                    s_594[5] = t_591[5];
                    s_594[6] = t_591[6];
                    s_594[7] = t_591[7];
                    s_594[8] = t_591[8];
                    s_594[9] = t_591[9];
                    s_594[10] = t_591[10];
                    s_594[11] = t_591[11];
                    s_594[12] = t_591[12];
                    s_594[13] = t_591[13];
                    s_594[14] = t_591[14];
                    s_594[15] = t_591[15];
                    if (fodd_592) {
                        s_594[0] = t_591[4];
                        s_594[4] = t_591[0];
                        s_594[8] = t_591[12];
                        s_594[12] = t_591[8];
                        s_594[1] = t_591[5];
                        s_594[5] = t_591[1];
                        s_594[9] = t_591[13];
                        s_594[13] = t_591[9];
                        s_594[2] = t_591[6];
                        s_594[6] = t_591[2];
                        s_594[10] = t_591[14];
                        s_594[14] = t_591[10];
                        s_594[3] = t_591[7];
                        s_594[7] = t_591[3];
                        s_594[11] = t_591[15];
                        s_594[15] = t_591[11];
                    }
                    unsigned int r_595[16];
                    r_595[0] = s_594[0];
                    r_595[1] = s_594[1];
                    r_595[2] = s_594[2];
                    r_595[3] = s_594[3];
                    r_595[4] = s_594[4];
                    r_595[5] = s_594[5];
                    r_595[6] = s_594[6];
                    r_595[7] = s_594[7];
                    r_595[8] = s_594[8];
                    r_595[9] = s_594[9];
                    r_595[10] = s_594[10];
                    r_595[11] = s_594[11];
                    r_595[12] = s_594[12];
                    r_595[13] = s_594[13];
                    r_595[14] = s_594[14];
                    r_595[15] = s_594[15];
                    if (fhi_593) {
                        r_595[0] = s_594[8];
                        r_595[4] = s_594[12];
                        r_595[8] = s_594[0];
                        r_595[12] = s_594[4];
                        r_595[1] = s_594[9];
                        r_595[5] = s_594[13];
                        r_595[9] = s_594[1];
                        r_595[13] = s_594[5];
                        r_595[2] = s_594[10];
                        r_595[6] = s_594[14];
                        r_595[10] = s_594[2];
                        r_595[14] = s_594[6];
                        r_595[3] = s_594[11];
                        r_595[7] = s_594[15];
                        r_595[11] = s_594[3];
                        r_595[15] = s_594[7];
                    }
                    float2 a2_596 = make_float2(acc_490[0], acc_490[1]);
                    float2 z_597 = add_f32x2_noftz(a2_596, make_float2(__uint_as_float(r_595[0] << 16), __uint_as_float(r_595[0] & 4294901760u)));
                    acc_490[0] = z_597.x;
                    acc_490[1] = z_597.y;
                    float2 ss2_598 = fma_f32x2_rn_noftz(z_597, z_597, make_float2(ssx, ssy));
                    ssx = ss2_598.x;
                    ssy = ss2_598.y;
                    float2 a2_599 = make_float2(acc_490[2], acc_490[3]);
                    float2 z_600 = add_f32x2_noftz(a2_599, make_float2(__uint_as_float(r_595[1] << 16), __uint_as_float(r_595[1] & 4294901760u)));
                    acc_490[2] = z_600.x;
                    acc_490[3] = z_600.y;
                    float2 ss2_601 = fma_f32x2_rn_noftz(z_600, z_600, make_float2(ssx, ssy));
                    ssx = ss2_601.x;
                    ssy = ss2_601.y;
                    float2 a2_602 = make_float2(acc_490[4], acc_490[5]);
                    float2 z_603 = add_f32x2_noftz(a2_602, make_float2(__uint_as_float(r_595[2] << 16), __uint_as_float(r_595[2] & 4294901760u)));
                    acc_490[4] = z_603.x;
                    acc_490[5] = z_603.y;
                    float2 ss2_604 = fma_f32x2_rn_noftz(z_603, z_603, make_float2(ssx, ssy));
                    ssx = ss2_604.x;
                    ssy = ss2_604.y;
                    float2 a2_605 = make_float2(acc_490[6], acc_490[7]);
                    float2 z_606 = add_f32x2_noftz(a2_605, make_float2(__uint_as_float(r_595[3] << 16), __uint_as_float(r_595[3] & 4294901760u)));
                    acc_490[6] = z_606.x;
                    acc_490[7] = z_606.y;
                    float2 ss2_607 = fma_f32x2_rn_noftz(z_606, z_606, make_float2(ssx, ssy));
                    ssx = ss2_607.x;
                    ssy = ss2_607.y;
                    float2 a2_608 = make_float2(acc_490[8], acc_490[9]);
                    float2 z_609 = add_f32x2_noftz(a2_608, make_float2(__uint_as_float(r_595[4] << 16), __uint_as_float(r_595[4] & 4294901760u)));
                    acc_490[8] = z_609.x;
                    acc_490[9] = z_609.y;
                    float2 ss2_610 = fma_f32x2_rn_noftz(z_609, z_609, make_float2(ssx, ssy));
                    ssx = ss2_610.x;
                    ssy = ss2_610.y;
                    float2 a2_611 = make_float2(acc_490[10], acc_490[11]);
                    float2 z_612 = add_f32x2_noftz(a2_611, make_float2(__uint_as_float(r_595[5] << 16), __uint_as_float(r_595[5] & 4294901760u)));
                    acc_490[10] = z_612.x;
                    acc_490[11] = z_612.y;
                    float2 ss2_613 = fma_f32x2_rn_noftz(z_612, z_612, make_float2(ssx, ssy));
                    ssx = ss2_613.x;
                    ssy = ss2_613.y;
                    float2 a2_614 = make_float2(acc_490[12], acc_490[13]);
                    float2 z_615 = add_f32x2_noftz(a2_614, make_float2(__uint_as_float(r_595[6] << 16), __uint_as_float(r_595[6] & 4294901760u)));
                    acc_490[12] = z_615.x;
                    acc_490[13] = z_615.y;
                    float2 ss2_616 = fma_f32x2_rn_noftz(z_615, z_615, make_float2(ssx, ssy));
                    ssx = ss2_616.x;
                    ssy = ss2_616.y;
                    float2 a2_617 = make_float2(acc_490[14], acc_490[15]);
                    float2 z_618 = add_f32x2_noftz(a2_617, make_float2(__uint_as_float(r_595[7] << 16), __uint_as_float(r_595[7] & 4294901760u)));
                    acc_490[14] = z_618.x;
                    acc_490[15] = z_618.y;
                    float2 ss2_619 = fma_f32x2_rn_noftz(z_618, z_618, make_float2(ssx, ssy));
                    ssx = ss2_619.x;
                    ssy = ss2_619.y;
                    float2 a2_620 = make_float2(acc_490[16], acc_490[17]);
                    float2 z_621 = add_f32x2_noftz(a2_620, make_float2(__uint_as_float(r_595[8] << 16), __uint_as_float(r_595[8] & 4294901760u)));
                    acc_490[16] = z_621.x;
                    acc_490[17] = z_621.y;
                    float2 ss2_622 = fma_f32x2_rn_noftz(z_621, z_621, make_float2(ssx, ssy));
                    ssx = ss2_622.x;
                    ssy = ss2_622.y;
                    float2 a2_623 = make_float2(acc_490[18], acc_490[19]);
                    float2 z_624 = add_f32x2_noftz(a2_623, make_float2(__uint_as_float(r_595[9] << 16), __uint_as_float(r_595[9] & 4294901760u)));
                    acc_490[18] = z_624.x;
                    acc_490[19] = z_624.y;
                    float2 ss2_625 = fma_f32x2_rn_noftz(z_624, z_624, make_float2(ssx, ssy));
                    ssx = ss2_625.x;
                    ssy = ss2_625.y;
                    float2 a2_626 = make_float2(acc_490[20], acc_490[21]);
                    float2 z_627 = add_f32x2_noftz(a2_626, make_float2(__uint_as_float(r_595[10] << 16), __uint_as_float(r_595[10] & 4294901760u)));
                    acc_490[20] = z_627.x;
                    acc_490[21] = z_627.y;
                    float2 ss2_628 = fma_f32x2_rn_noftz(z_627, z_627, make_float2(ssx, ssy));
                    ssx = ss2_628.x;
                    ssy = ss2_628.y;
                    float2 a2_629 = make_float2(acc_490[22], acc_490[23]);
                    float2 z_630 = add_f32x2_noftz(a2_629, make_float2(__uint_as_float(r_595[11] << 16), __uint_as_float(r_595[11] & 4294901760u)));
                    acc_490[22] = z_630.x;
                    acc_490[23] = z_630.y;
                    float2 ss2_631 = fma_f32x2_rn_noftz(z_630, z_630, make_float2(ssx, ssy));
                    ssx = ss2_631.x;
                    ssy = ss2_631.y;
                    float2 a2_632 = make_float2(acc_490[24], acc_490[25]);
                    float2 z_633 = add_f32x2_noftz(a2_632, make_float2(__uint_as_float(r_595[12] << 16), __uint_as_float(r_595[12] & 4294901760u)));
                    acc_490[24] = z_633.x;
                    acc_490[25] = z_633.y;
                    float2 ss2_634 = fma_f32x2_rn_noftz(z_633, z_633, make_float2(ssx, ssy));
                    ssx = ss2_634.x;
                    ssy = ss2_634.y;
                    float2 a2_635 = make_float2(acc_490[26], acc_490[27]);
                    float2 z_636 = add_f32x2_noftz(a2_635, make_float2(__uint_as_float(r_595[13] << 16), __uint_as_float(r_595[13] & 4294901760u)));
                    acc_490[26] = z_636.x;
                    acc_490[27] = z_636.y;
                    float2 ss2_637 = fma_f32x2_rn_noftz(z_636, z_636, make_float2(ssx, ssy));
                    ssx = ss2_637.x;
                    ssy = ss2_637.y;
                    float2 a2_638 = make_float2(acc_490[28], acc_490[29]);
                    float2 z_639 = add_f32x2_noftz(a2_638, make_float2(__uint_as_float(r_595[14] << 16), __uint_as_float(r_595[14] & 4294901760u)));
                    acc_490[28] = z_639.x;
                    acc_490[29] = z_639.y;
                    float2 ss2_640 = fma_f32x2_rn_noftz(z_639, z_639, make_float2(ssx, ssy));
                    ssx = ss2_640.x;
                    ssy = ss2_640.y;
                    float2 a2_641 = make_float2(acc_490[30], acc_490[31]);
                    float2 z_642 = add_f32x2_noftz(a2_641, make_float2(__uint_as_float(r_595[15] << 16), __uint_as_float(r_595[15] & 4294901760u)));
                    acc_490[30] = z_642.x;
                    acc_490[31] = z_642.y;
                    float2 ss2_643 = fma_f32x2_rn_noftz(z_642, z_642, make_float2(ssx, ssy));
                    ssx = ss2_643.x;
                    ssy = ss2_643.y;
                    unsigned int acc_490_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_490[_lp*2 + 0], acc_490[_lp*2+1 + 0]));
                        acc_490_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_644 = (f & 1) != 0;
                    bool fhi_645 = (f & 2) != 0;
                    unsigned int s_646[16];
                    s_646[0] = acc_490_bf16[0];
                    s_646[1] = acc_490_bf16[1];
                    s_646[2] = acc_490_bf16[2];
                    s_646[3] = acc_490_bf16[3];
                    s_646[4] = acc_490_bf16[4];
                    s_646[5] = acc_490_bf16[5];
                    s_646[6] = acc_490_bf16[6];
                    s_646[7] = acc_490_bf16[7];
                    s_646[8] = acc_490_bf16[8];
                    s_646[9] = acc_490_bf16[9];
                    s_646[10] = acc_490_bf16[10];
                    s_646[11] = acc_490_bf16[11];
                    s_646[12] = acc_490_bf16[12];
                    s_646[13] = acc_490_bf16[13];
                    s_646[14] = acc_490_bf16[14];
                    s_646[15] = acc_490_bf16[15];
                    if (fodd_644) {
                        s_646[0] = acc_490_bf16[4];
                        s_646[4] = acc_490_bf16[0];
                        s_646[8] = acc_490_bf16[12];
                        s_646[12] = acc_490_bf16[8];
                        s_646[1] = acc_490_bf16[5];
                        s_646[5] = acc_490_bf16[1];
                        s_646[9] = acc_490_bf16[13];
                        s_646[13] = acc_490_bf16[9];
                        s_646[2] = acc_490_bf16[6];
                        s_646[6] = acc_490_bf16[2];
                        s_646[10] = acc_490_bf16[14];
                        s_646[14] = acc_490_bf16[10];
                        s_646[3] = acc_490_bf16[7];
                        s_646[7] = acc_490_bf16[3];
                        s_646[11] = acc_490_bf16[15];
                        s_646[15] = acc_490_bf16[11];
                    }
                    unsigned int r_647[16];
                    r_647[0] = s_646[0];
                    r_647[1] = s_646[1];
                    r_647[2] = s_646[2];
                    r_647[3] = s_646[3];
                    r_647[4] = s_646[4];
                    r_647[5] = s_646[5];
                    r_647[6] = s_646[6];
                    r_647[7] = s_646[7];
                    r_647[8] = s_646[8];
                    r_647[9] = s_646[9];
                    r_647[10] = s_646[10];
                    r_647[11] = s_646[11];
                    r_647[12] = s_646[12];
                    r_647[13] = s_646[13];
                    r_647[14] = s_646[14];
                    r_647[15] = s_646[15];
                    if (fhi_645) {
                        r_647[0] = s_646[8];
                        r_647[4] = s_646[12];
                        r_647[8] = s_646[0];
                        r_647[12] = s_646[4];
                        r_647[1] = s_646[9];
                        r_647[5] = s_646[13];
                        r_647[9] = s_646[1];
                        r_647[13] = s_646[5];
                        r_647[2] = s_646[10];
                        r_647[6] = s_646[14];
                        r_647[10] = s_646[2];
                        r_647[14] = s_646[6];
                        r_647[3] = s_646[11];
                        r_647[7] = s_646[15];
                        r_647[11] = s_646[3];
                        r_647[15] = s_646[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 0))), "r"(r_647[0]), "r"(r_647[1]), "r"(r_647[2]), "r"(r_647[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 1))), "r"(r_647[4]), "r"(r_647[5]), "r"(r_647[6]), "r"(r_647[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 2))), "r"(r_647[8]), "r"(r_647[9]), "r"(r_647[10]), "r"(r_647[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_483 + 16 * (f ^ 3))), "r"(r_647[12]), "r"(r_647[13]), "r"(r_647[14]), "r"(r_647[15]) : "memory");
                    unsigned int d_648 = 0;
                    d_648 = d_648 | pv[16] & ~((pv[16] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[17] & ~((pv[17] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[18] & ~((pv[18] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[19] & ~((pv[19] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | ((ps[4] & 255) + 1 | (ps[4] >> 8) + 1) & 256;
                    d_648 = d_648 | pv[20] & ~((pv[20] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[21] & ~((pv[21] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[22] & ~((pv[22] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | pv[23] & ~((pv[23] & 2004318071) + 2004318071) & 2290649224u;
                    d_648 = d_648 | ((ps[5] & 255) + 1 | (ps[5] >> 8) + 1) & 256;
                    if (d_648 != 0) {
                        int pp0_2 = lane_2 + 128;
                        int pp1_2 = pp0_2 + 32;
                        uint32_t _fp4_wire_values_2[8];
                        uint32_t _fp4_wire_scales_2[2];
                        uint16_t _wire_h16_7_0;
                        uint16_t _wire_h16_7_1;
                        uint32_t _wire_vpend_7 = 3u;
                        uint32_t _wire_spend_7 = 3u;
                        do {
                            if (_wire_vpend_7 & 1u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[0]), "=r"(_fp4_wire_values_2[1]), "=r"(_fp4_wire_values_2[2]), "=r"(_fp4_wire_values_2[3]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp0_2)))) : "memory");
                            }
                            if (_wire_vpend_7 & 2u) {
                                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_fp4_wire_values_2[4]), "=r"(_fp4_wire_values_2[5]), "=r"(_fp4_wire_values_2[6]), "=r"(_fp4_wire_values_2[7]) : "l"(reinterpret_cast<unsigned int*>(os_wire + (v_row + (long long)(16 * pp1_2)))) : "memory");
                            }
                            if (_wire_spend_7 & 1u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_7_0) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp0_2)))) : "memory");
                                _fp4_wire_scales_2[0] = (uint32_t)_wire_h16_7_0;
                            }
                            if (_wire_spend_7 & 2u) {
                                asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(_wire_h16_7_1) : "l"(reinterpret_cast<uint16_t*>(os_wire + (s_row + (long long)(2 * pp1_2)))) : "memory");
                                _fp4_wire_scales_2[1] = (uint32_t)_wire_h16_7_1;
                            }
                            _wire_vpend_7 = ((uint32_t)((((_fp4_wire_values_2[0] & ~((_fp4_wire_values_2[0] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[1] & ~((_fp4_wire_values_2[1] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[2] & ~((_fp4_wire_values_2[2] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[3] & ~((_fp4_wire_values_2[3] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 0) | ((uint32_t)((((_fp4_wire_values_2[4] & ~((_fp4_wire_values_2[4] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[5] & ~((_fp4_wire_values_2[5] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[6] & ~((_fp4_wire_values_2[6] & 0x77777777u) + 0x77777777u)) & 0x88888888u) | ((_fp4_wire_values_2[7] & ~((_fp4_wire_values_2[7] & 0x77777777u) + 0x77777777u)) & 0x88888888u)) != 0u) << 1);
                            _wire_spend_7 = ((uint32_t)(((uint32_t)((_fp4_wire_scales_2[0] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_2[0] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 0) | ((uint32_t)(((uint32_t)((_fp4_wire_scales_2[1] & 0xFFu) == 0xFFu) | (uint32_t)(((_fp4_wire_scales_2[1] >> 8) & 0xFFu) == 0xFFu)) != 0u) << 1);
                        } while ((_wire_vpend_7 | _wire_spend_7) != 0u);
                        pv[16] = _fp4_wire_values_2[0];
                        pv[17] = _fp4_wire_values_2[1];
                        pv[18] = _fp4_wire_values_2[2];
                        pv[19] = _fp4_wire_values_2[3];
                        ps[4] = _fp4_wire_scales_2[0];
                        pv[20] = _fp4_wire_values_2[4];
                        pv[21] = _fp4_wire_values_2[5];
                        pv[22] = _fp4_wire_values_2[6];
                        pv[23] = _fp4_wire_values_2[7];
                        ps[5] = _fp4_wire_scales_2[1];
                    }
                    int pp_649 = lane_2 + 128;
                    unsigned int own_650[4];
                    unsigned int own_sc_651[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_650[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_650[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_650[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_650[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_649)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_651[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_649)));
                    unsigned int a_652[4];
                    unsigned int b_653[4];
                    a_652[0] = ((first) ? own_650[0] : pv[16]);
                    b_653[0] = ((first) ? pv[16] : own_650[0]);
                    a_652[1] = ((first) ? own_650[1] : pv[17]);
                    b_653[1] = ((first) ? pv[17] : own_650[1]);
                    a_652[2] = ((first) ? own_650[2] : pv[18]);
                    b_653[2] = ((first) ? pv[18] : own_650[2]);
                    a_652[3] = ((first) ? own_650[3] : pv[19]);
                    b_653[3] = ((first) ? pv[19] : own_650[3]);
                    unsigned int a_sc_654 = ((first) ? own_sc_651[0] : ps[4]);
                    unsigned int b_sc_655 = ((first) ? ps[4] : own_sc_651[0]);
                    float acc_656[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(a_sc_654)));
                    float _cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                    float _cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_16), "f"(rs_a_f));
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_17), "f"(rs_a_f));
                    float2 scale0_657 = make_float2(_mul_16, _mul_16);
                    float2 scale1_658 = make_float2(_mul_17, _mul_17);
                    float _unpack_fp4_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_652[0]));
                        (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(a_652[0]));
                        #endif
                    }
                    float2 a_659 = make_float2(acc_656[0], acc_656[1]);
                    float2 q_660 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                    float2 r_661 = fma_f32x2_rn_noftz(q_660, scale0_657, a_659);
                    acc_656[0] = r_661.x;
                    acc_656[1] = r_661.y;
                    float2 a_662 = make_float2(acc_656[2], acc_656[3]);
                    float2 q_663 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                    float2 r_664 = fma_f32x2_rn_noftz(q_663, scale0_657, a_662);
                    acc_656[2] = r_664.x;
                    acc_656[3] = r_664.y;
                    float2 a_665 = make_float2(acc_656[4], acc_656[5]);
                    float2 q_666 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                    float2 r_667 = fma_f32x2_rn_noftz(q_666, scale0_657, a_665);
                    acc_656[4] = r_667.x;
                    acc_656[5] = r_667.y;
                    float2 a_668 = make_float2(acc_656[6], acc_656[7]);
                    float2 q_669 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                    float2 r_670 = fma_f32x2_rn_noftz(q_669, scale0_657, a_668);
                    acc_656[6] = r_670.x;
                    acc_656[7] = r_670.y;
                    float _unpack_fp4_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_652[1 + 0]));
                        (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(a_652[1 + 0]));
                        #endif
                    }
                    float2 a_671 = make_float2(acc_656[8], acc_656[9]);
                    float2 q_672 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                    float2 r_673 = fma_f32x2_rn_noftz(q_672, scale0_657, a_671);
                    acc_656[8] = r_673.x;
                    acc_656[9] = r_673.y;
                    float2 a_674 = make_float2(acc_656[10], acc_656[11]);
                    float2 q_675 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                    float2 r_676 = fma_f32x2_rn_noftz(q_675, scale0_657, a_674);
                    acc_656[10] = r_676.x;
                    acc_656[11] = r_676.y;
                    float2 a_677 = make_float2(acc_656[12], acc_656[13]);
                    float2 q_678 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                    float2 r_679 = fma_f32x2_rn_noftz(q_678, scale0_657, a_677);
                    acc_656[12] = r_679.x;
                    acc_656[13] = r_679.y;
                    float2 a_680 = make_float2(acc_656[14], acc_656[15]);
                    float2 q_681 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                    float2 r_682 = fma_f32x2_rn_noftz(q_681, scale0_657, a_680);
                    acc_656[14] = r_682.x;
                    acc_656[15] = r_682.y;
                    float _unpack_fp4_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_652[2 + 0]));
                        (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(a_652[2 + 0]));
                        #endif
                    }
                    float2 a_683 = make_float2(acc_656[16], acc_656[17]);
                    float2 q_684 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                    float2 r_685 = fma_f32x2_rn_noftz(q_684, scale1_658, a_683);
                    acc_656[16] = r_685.x;
                    acc_656[17] = r_685.y;
                    float2 a_686 = make_float2(acc_656[18], acc_656[19]);
                    float2 q_687 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                    float2 r_688 = fma_f32x2_rn_noftz(q_687, scale1_658, a_686);
                    acc_656[18] = r_688.x;
                    acc_656[19] = r_688.y;
                    float2 a_689 = make_float2(acc_656[20], acc_656[21]);
                    float2 q_690 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                    float2 r_691 = fma_f32x2_rn_noftz(q_690, scale1_658, a_689);
                    acc_656[20] = r_691.x;
                    acc_656[21] = r_691.y;
                    float2 a_692 = make_float2(acc_656[22], acc_656[23]);
                    float2 q_693 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                    float2 r_694 = fma_f32x2_rn_noftz(q_693, scale1_658, a_692);
                    acc_656[22] = r_694.x;
                    acc_656[23] = r_694.y;
                    float _unpack_fp4_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_652[3 + 0]));
                        (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(a_652[3 + 0]));
                        #endif
                    }
                    float2 a_695 = make_float2(acc_656[24], acc_656[25]);
                    float2 q_696 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                    float2 r_697 = fma_f32x2_rn_noftz(q_696, scale1_658, a_695);
                    acc_656[24] = r_697.x;
                    acc_656[25] = r_697.y;
                    float2 a_698 = make_float2(acc_656[26], acc_656[27]);
                    float2 q_699 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                    float2 r_700 = fma_f32x2_rn_noftz(q_699, scale1_658, a_698);
                    acc_656[26] = r_700.x;
                    acc_656[27] = r_700.y;
                    float2 a_701 = make_float2(acc_656[28], acc_656[29]);
                    float2 q_702 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                    float2 r_703 = fma_f32x2_rn_noftz(q_702, scale1_658, a_701);
                    acc_656[28] = r_703.x;
                    acc_656[29] = r_703.y;
                    float2 a_704 = make_float2(acc_656[30], acc_656[31]);
                    float2 q_705 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                    float2 r_706 = fma_f32x2_rn_noftz(q_705, scale1_658, a_704);
                    acc_656[30] = r_706.x;
                    acc_656[31] = r_706.y;
                    uint32_t _e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(b_sc_655)));
                    float _cvt_f32_f16_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                    float _cvt_f32_f16_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                    float _mul_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_18), "f"(rs_b_f));
                    float _mul_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_19), "f"(rs_b_f));
                    float2 scale0_707 = make_float2(_mul_18, _mul_18);
                    float2 scale1_708 = make_float2(_mul_19, _mul_19);
                    float _unpack_fp4_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_653[0]));
                        (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(b_653[0]));
                        #endif
                    }
                    float2 a_709 = make_float2(acc_656[0], acc_656[1]);
                    float2 q_710 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                    float2 r_711 = fma_f32x2_rn_noftz(q_710, scale0_707, a_709);
                    acc_656[0] = r_711.x;
                    acc_656[1] = r_711.y;
                    float2 a_712 = make_float2(acc_656[2], acc_656[3]);
                    float2 q_713 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                    float2 r_714 = fma_f32x2_rn_noftz(q_713, scale0_707, a_712);
                    acc_656[2] = r_714.x;
                    acc_656[3] = r_714.y;
                    float2 a_715 = make_float2(acc_656[4], acc_656[5]);
                    float2 q_716 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                    float2 r_717 = fma_f32x2_rn_noftz(q_716, scale0_707, a_715);
                    acc_656[4] = r_717.x;
                    acc_656[5] = r_717.y;
                    float2 a_718 = make_float2(acc_656[6], acc_656[7]);
                    float2 q_719 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                    float2 r_720 = fma_f32x2_rn_noftz(q_719, scale0_707, a_718);
                    acc_656[6] = r_720.x;
                    acc_656[7] = r_720.y;
                    float _unpack_fp4_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_653[1 + 0]));
                        (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(b_653[1 + 0]));
                        #endif
                    }
                    float2 a_721 = make_float2(acc_656[8], acc_656[9]);
                    float2 q_722 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                    float2 r_723 = fma_f32x2_rn_noftz(q_722, scale0_707, a_721);
                    acc_656[8] = r_723.x;
                    acc_656[9] = r_723.y;
                    float2 a_724 = make_float2(acc_656[10], acc_656[11]);
                    float2 q_725 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                    float2 r_726 = fma_f32x2_rn_noftz(q_725, scale0_707, a_724);
                    acc_656[10] = r_726.x;
                    acc_656[11] = r_726.y;
                    float2 a_727 = make_float2(acc_656[12], acc_656[13]);
                    float2 q_728 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                    float2 r_729 = fma_f32x2_rn_noftz(q_728, scale0_707, a_727);
                    acc_656[12] = r_729.x;
                    acc_656[13] = r_729.y;
                    float2 a_730 = make_float2(acc_656[14], acc_656[15]);
                    float2 q_731 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                    float2 r_732 = fma_f32x2_rn_noftz(q_731, scale0_707, a_730);
                    acc_656[14] = r_732.x;
                    acc_656[15] = r_732.y;
                    float _unpack_fp4_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_653[2 + 0]));
                        (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(b_653[2 + 0]));
                        #endif
                    }
                    float2 a_733 = make_float2(acc_656[16], acc_656[17]);
                    float2 q_734 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                    float2 r_735 = fma_f32x2_rn_noftz(q_734, scale1_708, a_733);
                    acc_656[16] = r_735.x;
                    acc_656[17] = r_735.y;
                    float2 a_736 = make_float2(acc_656[18], acc_656[19]);
                    float2 q_737 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                    float2 r_738 = fma_f32x2_rn_noftz(q_737, scale1_708, a_736);
                    acc_656[18] = r_738.x;
                    acc_656[19] = r_738.y;
                    float2 a_739 = make_float2(acc_656[20], acc_656[21]);
                    float2 q_740 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                    float2 r_741 = fma_f32x2_rn_noftz(q_740, scale1_708, a_739);
                    acc_656[20] = r_741.x;
                    acc_656[21] = r_741.y;
                    float2 a_742 = make_float2(acc_656[22], acc_656[23]);
                    float2 q_743 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                    float2 r_744 = fma_f32x2_rn_noftz(q_743, scale1_708, a_742);
                    acc_656[22] = r_744.x;
                    acc_656[23] = r_744.y;
                    float _unpack_fp4_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_653[3 + 0]));
                        (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(b_653[3 + 0]));
                        #endif
                    }
                    float2 a_745 = make_float2(acc_656[24], acc_656[25]);
                    float2 q_746 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                    float2 r_747 = fma_f32x2_rn_noftz(q_746, scale1_708, a_745);
                    acc_656[24] = r_747.x;
                    acc_656[25] = r_747.y;
                    float2 a_748 = make_float2(acc_656[26], acc_656[27]);
                    float2 q_749 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                    float2 r_750 = fma_f32x2_rn_noftz(q_749, scale1_708, a_748);
                    acc_656[26] = r_750.x;
                    acc_656[27] = r_750.y;
                    float2 a_751 = make_float2(acc_656[28], acc_656[29]);
                    float2 q_752 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                    float2 r_753 = fma_f32x2_rn_noftz(q_752, scale1_708, a_751);
                    acc_656[28] = r_753.x;
                    acc_656[29] = r_753.y;
                    float2 a_754 = make_float2(acc_656[30], acc_656[31]);
                    float2 q_755 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                    float2 r_756 = fma_f32x2_rn_noftz(q_755, scale1_708, a_754);
                    acc_656[30] = r_756.x;
                    acc_656[31] = r_756.y;
                    unsigned int t_757[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_757[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_757[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_757[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_757[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_757[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 3))));
                    bool fodd_758 = (f & 1) != 0;
                    bool fhi_759 = (f & 2) != 0;
                    unsigned int s_760[16];
                    s_760[0] = t_757[0];
                    s_760[1] = t_757[1];
                    s_760[2] = t_757[2];
                    s_760[3] = t_757[3];
                    s_760[4] = t_757[4];
                    s_760[5] = t_757[5];
                    s_760[6] = t_757[6];
                    s_760[7] = t_757[7];
                    s_760[8] = t_757[8];
                    s_760[9] = t_757[9];
                    s_760[10] = t_757[10];
                    s_760[11] = t_757[11];
                    s_760[12] = t_757[12];
                    s_760[13] = t_757[13];
                    s_760[14] = t_757[14];
                    s_760[15] = t_757[15];
                    if (fodd_758) {
                        s_760[0] = t_757[4];
                        s_760[4] = t_757[0];
                        s_760[8] = t_757[12];
                        s_760[12] = t_757[8];
                        s_760[1] = t_757[5];
                        s_760[5] = t_757[1];
                        s_760[9] = t_757[13];
                        s_760[13] = t_757[9];
                        s_760[2] = t_757[6];
                        s_760[6] = t_757[2];
                        s_760[10] = t_757[14];
                        s_760[14] = t_757[10];
                        s_760[3] = t_757[7];
                        s_760[7] = t_757[3];
                        s_760[11] = t_757[15];
                        s_760[15] = t_757[11];
                    }
                    unsigned int r_761[16];
                    r_761[0] = s_760[0];
                    r_761[1] = s_760[1];
                    r_761[2] = s_760[2];
                    r_761[3] = s_760[3];
                    r_761[4] = s_760[4];
                    r_761[5] = s_760[5];
                    r_761[6] = s_760[6];
                    r_761[7] = s_760[7];
                    r_761[8] = s_760[8];
                    r_761[9] = s_760[9];
                    r_761[10] = s_760[10];
                    r_761[11] = s_760[11];
                    r_761[12] = s_760[12];
                    r_761[13] = s_760[13];
                    r_761[14] = s_760[14];
                    r_761[15] = s_760[15];
                    if (fhi_759) {
                        r_761[0] = s_760[8];
                        r_761[4] = s_760[12];
                        r_761[8] = s_760[0];
                        r_761[12] = s_760[4];
                        r_761[1] = s_760[9];
                        r_761[5] = s_760[13];
                        r_761[9] = s_760[1];
                        r_761[13] = s_760[5];
                        r_761[2] = s_760[10];
                        r_761[6] = s_760[14];
                        r_761[10] = s_760[2];
                        r_761[14] = s_760[6];
                        r_761[3] = s_760[11];
                        r_761[7] = s_760[15];
                        r_761[11] = s_760[3];
                        r_761[15] = s_760[7];
                    }
                    float2 a2_762 = make_float2(acc_656[0], acc_656[1]);
                    float2 z_763 = add_f32x2_noftz(a2_762, make_float2(__uint_as_float(r_761[0] << 16), __uint_as_float(r_761[0] & 4294901760u)));
                    acc_656[0] = z_763.x;
                    acc_656[1] = z_763.y;
                    float2 ss2_764 = fma_f32x2_rn_noftz(z_763, z_763, make_float2(ssx, ssy));
                    ssx = ss2_764.x;
                    ssy = ss2_764.y;
                    float2 a2_765 = make_float2(acc_656[2], acc_656[3]);
                    float2 z_766 = add_f32x2_noftz(a2_765, make_float2(__uint_as_float(r_761[1] << 16), __uint_as_float(r_761[1] & 4294901760u)));
                    acc_656[2] = z_766.x;
                    acc_656[3] = z_766.y;
                    float2 ss2_767 = fma_f32x2_rn_noftz(z_766, z_766, make_float2(ssx, ssy));
                    ssx = ss2_767.x;
                    ssy = ss2_767.y;
                    float2 a2_768 = make_float2(acc_656[4], acc_656[5]);
                    float2 z_769 = add_f32x2_noftz(a2_768, make_float2(__uint_as_float(r_761[2] << 16), __uint_as_float(r_761[2] & 4294901760u)));
                    acc_656[4] = z_769.x;
                    acc_656[5] = z_769.y;
                    float2 ss2_770 = fma_f32x2_rn_noftz(z_769, z_769, make_float2(ssx, ssy));
                    ssx = ss2_770.x;
                    ssy = ss2_770.y;
                    float2 a2_771 = make_float2(acc_656[6], acc_656[7]);
                    float2 z_772 = add_f32x2_noftz(a2_771, make_float2(__uint_as_float(r_761[3] << 16), __uint_as_float(r_761[3] & 4294901760u)));
                    acc_656[6] = z_772.x;
                    acc_656[7] = z_772.y;
                    float2 ss2_773 = fma_f32x2_rn_noftz(z_772, z_772, make_float2(ssx, ssy));
                    ssx = ss2_773.x;
                    ssy = ss2_773.y;
                    float2 a2_774 = make_float2(acc_656[8], acc_656[9]);
                    float2 z_775 = add_f32x2_noftz(a2_774, make_float2(__uint_as_float(r_761[4] << 16), __uint_as_float(r_761[4] & 4294901760u)));
                    acc_656[8] = z_775.x;
                    acc_656[9] = z_775.y;
                    float2 ss2_776 = fma_f32x2_rn_noftz(z_775, z_775, make_float2(ssx, ssy));
                    ssx = ss2_776.x;
                    ssy = ss2_776.y;
                    float2 a2_777 = make_float2(acc_656[10], acc_656[11]);
                    float2 z_778 = add_f32x2_noftz(a2_777, make_float2(__uint_as_float(r_761[5] << 16), __uint_as_float(r_761[5] & 4294901760u)));
                    acc_656[10] = z_778.x;
                    acc_656[11] = z_778.y;
                    float2 ss2_779 = fma_f32x2_rn_noftz(z_778, z_778, make_float2(ssx, ssy));
                    ssx = ss2_779.x;
                    ssy = ss2_779.y;
                    float2 a2_780 = make_float2(acc_656[12], acc_656[13]);
                    float2 z_781 = add_f32x2_noftz(a2_780, make_float2(__uint_as_float(r_761[6] << 16), __uint_as_float(r_761[6] & 4294901760u)));
                    acc_656[12] = z_781.x;
                    acc_656[13] = z_781.y;
                    float2 ss2_782 = fma_f32x2_rn_noftz(z_781, z_781, make_float2(ssx, ssy));
                    ssx = ss2_782.x;
                    ssy = ss2_782.y;
                    float2 a2_783 = make_float2(acc_656[14], acc_656[15]);
                    float2 z_784 = add_f32x2_noftz(a2_783, make_float2(__uint_as_float(r_761[7] << 16), __uint_as_float(r_761[7] & 4294901760u)));
                    acc_656[14] = z_784.x;
                    acc_656[15] = z_784.y;
                    float2 ss2_785 = fma_f32x2_rn_noftz(z_784, z_784, make_float2(ssx, ssy));
                    ssx = ss2_785.x;
                    ssy = ss2_785.y;
                    float2 a2_786 = make_float2(acc_656[16], acc_656[17]);
                    float2 z_787 = add_f32x2_noftz(a2_786, make_float2(__uint_as_float(r_761[8] << 16), __uint_as_float(r_761[8] & 4294901760u)));
                    acc_656[16] = z_787.x;
                    acc_656[17] = z_787.y;
                    float2 ss2_788 = fma_f32x2_rn_noftz(z_787, z_787, make_float2(ssx, ssy));
                    ssx = ss2_788.x;
                    ssy = ss2_788.y;
                    float2 a2_789 = make_float2(acc_656[18], acc_656[19]);
                    float2 z_790 = add_f32x2_noftz(a2_789, make_float2(__uint_as_float(r_761[9] << 16), __uint_as_float(r_761[9] & 4294901760u)));
                    acc_656[18] = z_790.x;
                    acc_656[19] = z_790.y;
                    float2 ss2_791 = fma_f32x2_rn_noftz(z_790, z_790, make_float2(ssx, ssy));
                    ssx = ss2_791.x;
                    ssy = ss2_791.y;
                    float2 a2_792 = make_float2(acc_656[20], acc_656[21]);
                    float2 z_793 = add_f32x2_noftz(a2_792, make_float2(__uint_as_float(r_761[10] << 16), __uint_as_float(r_761[10] & 4294901760u)));
                    acc_656[20] = z_793.x;
                    acc_656[21] = z_793.y;
                    float2 ss2_794 = fma_f32x2_rn_noftz(z_793, z_793, make_float2(ssx, ssy));
                    ssx = ss2_794.x;
                    ssy = ss2_794.y;
                    float2 a2_795 = make_float2(acc_656[22], acc_656[23]);
                    float2 z_796 = add_f32x2_noftz(a2_795, make_float2(__uint_as_float(r_761[11] << 16), __uint_as_float(r_761[11] & 4294901760u)));
                    acc_656[22] = z_796.x;
                    acc_656[23] = z_796.y;
                    float2 ss2_797 = fma_f32x2_rn_noftz(z_796, z_796, make_float2(ssx, ssy));
                    ssx = ss2_797.x;
                    ssy = ss2_797.y;
                    float2 a2_798 = make_float2(acc_656[24], acc_656[25]);
                    float2 z_799 = add_f32x2_noftz(a2_798, make_float2(__uint_as_float(r_761[12] << 16), __uint_as_float(r_761[12] & 4294901760u)));
                    acc_656[24] = z_799.x;
                    acc_656[25] = z_799.y;
                    float2 ss2_800 = fma_f32x2_rn_noftz(z_799, z_799, make_float2(ssx, ssy));
                    ssx = ss2_800.x;
                    ssy = ss2_800.y;
                    float2 a2_801 = make_float2(acc_656[26], acc_656[27]);
                    float2 z_802 = add_f32x2_noftz(a2_801, make_float2(__uint_as_float(r_761[13] << 16), __uint_as_float(r_761[13] & 4294901760u)));
                    acc_656[26] = z_802.x;
                    acc_656[27] = z_802.y;
                    float2 ss2_803 = fma_f32x2_rn_noftz(z_802, z_802, make_float2(ssx, ssy));
                    ssx = ss2_803.x;
                    ssy = ss2_803.y;
                    float2 a2_804 = make_float2(acc_656[28], acc_656[29]);
                    float2 z_805 = add_f32x2_noftz(a2_804, make_float2(__uint_as_float(r_761[14] << 16), __uint_as_float(r_761[14] & 4294901760u)));
                    acc_656[28] = z_805.x;
                    acc_656[29] = z_805.y;
                    float2 ss2_806 = fma_f32x2_rn_noftz(z_805, z_805, make_float2(ssx, ssy));
                    ssx = ss2_806.x;
                    ssy = ss2_806.y;
                    float2 a2_807 = make_float2(acc_656[30], acc_656[31]);
                    float2 z_808 = add_f32x2_noftz(a2_807, make_float2(__uint_as_float(r_761[15] << 16), __uint_as_float(r_761[15] & 4294901760u)));
                    acc_656[30] = z_808.x;
                    acc_656[31] = z_808.y;
                    float2 ss2_809 = fma_f32x2_rn_noftz(z_808, z_808, make_float2(ssx, ssy));
                    ssx = ss2_809.x;
                    ssy = ss2_809.y;
                    unsigned int acc_656_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_656[_lp*2 + 0], acc_656[_lp*2+1 + 0]));
                        acc_656_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_810 = (f & 1) != 0;
                    bool fhi_811 = (f & 2) != 0;
                    unsigned int s_812[16];
                    s_812[0] = acc_656_bf16[0];
                    s_812[1] = acc_656_bf16[1];
                    s_812[2] = acc_656_bf16[2];
                    s_812[3] = acc_656_bf16[3];
                    s_812[4] = acc_656_bf16[4];
                    s_812[5] = acc_656_bf16[5];
                    s_812[6] = acc_656_bf16[6];
                    s_812[7] = acc_656_bf16[7];
                    s_812[8] = acc_656_bf16[8];
                    s_812[9] = acc_656_bf16[9];
                    s_812[10] = acc_656_bf16[10];
                    s_812[11] = acc_656_bf16[11];
                    s_812[12] = acc_656_bf16[12];
                    s_812[13] = acc_656_bf16[13];
                    s_812[14] = acc_656_bf16[14];
                    s_812[15] = acc_656_bf16[15];
                    if (fodd_810) {
                        s_812[0] = acc_656_bf16[4];
                        s_812[4] = acc_656_bf16[0];
                        s_812[8] = acc_656_bf16[12];
                        s_812[12] = acc_656_bf16[8];
                        s_812[1] = acc_656_bf16[5];
                        s_812[5] = acc_656_bf16[1];
                        s_812[9] = acc_656_bf16[13];
                        s_812[13] = acc_656_bf16[9];
                        s_812[2] = acc_656_bf16[6];
                        s_812[6] = acc_656_bf16[2];
                        s_812[10] = acc_656_bf16[14];
                        s_812[14] = acc_656_bf16[10];
                        s_812[3] = acc_656_bf16[7];
                        s_812[7] = acc_656_bf16[3];
                        s_812[11] = acc_656_bf16[15];
                        s_812[15] = acc_656_bf16[11];
                    }
                    unsigned int r_813[16];
                    r_813[0] = s_812[0];
                    r_813[1] = s_812[1];
                    r_813[2] = s_812[2];
                    r_813[3] = s_812[3];
                    r_813[4] = s_812[4];
                    r_813[5] = s_812[5];
                    r_813[6] = s_812[6];
                    r_813[7] = s_812[7];
                    r_813[8] = s_812[8];
                    r_813[9] = s_812[9];
                    r_813[10] = s_812[10];
                    r_813[11] = s_812[11];
                    r_813[12] = s_812[12];
                    r_813[13] = s_812[13];
                    r_813[14] = s_812[14];
                    r_813[15] = s_812[15];
                    if (fhi_811) {
                        r_813[0] = s_812[8];
                        r_813[4] = s_812[12];
                        r_813[8] = s_812[0];
                        r_813[12] = s_812[4];
                        r_813[1] = s_812[9];
                        r_813[5] = s_812[13];
                        r_813[9] = s_812[1];
                        r_813[13] = s_812[5];
                        r_813[2] = s_812[10];
                        r_813[6] = s_812[14];
                        r_813[10] = s_812[2];
                        r_813[14] = s_812[6];
                        r_813[3] = s_812[11];
                        r_813[7] = s_812[15];
                        r_813[11] = s_812[3];
                        r_813[15] = s_812[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 0))), "r"(r_813[0]), "r"(r_813[1]), "r"(r_813[2]), "r"(r_813[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 1))), "r"(r_813[4]), "r"(r_813[5]), "r"(r_813[6]), "r"(r_813[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 2))), "r"(r_813[8]), "r"(r_813[9]), "r"(r_813[10]), "r"(r_813[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_649 + 16 * (f ^ 3))), "r"(r_813[12]), "r"(r_813[13]), "r"(r_813[14]), "r"(r_813[15]) : "memory");
                    int pp_814 = lane_2 + 160;
                    unsigned int own_815[4];
                    unsigned int own_sc_816[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_815[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_815[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_815[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_815[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_814)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_816[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_814)));
                    unsigned int a_817[4];
                    unsigned int b_818[4];
                    a_817[0] = ((first) ? own_815[0] : pv[20]);
                    b_818[0] = ((first) ? pv[20] : own_815[0]);
                    a_817[1] = ((first) ? own_815[1] : pv[21]);
                    b_818[1] = ((first) ? pv[21] : own_815[1]);
                    a_817[2] = ((first) ? own_815[2] : pv[22]);
                    b_818[2] = ((first) ? pv[22] : own_815[2]);
                    a_817[3] = ((first) ? own_815[3] : pv[23]);
                    b_818[3] = ((first) ? pv[23] : own_815[3]);
                    unsigned int a_sc_819 = ((first) ? own_sc_816[0] : ps[5]);
                    unsigned int b_sc_820 = ((first) ? ps[5] : own_sc_816[0]);
                    float acc_821[32] = {0};
                    uint32_t _e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(a_sc_819)));
                    float _cvt_f32_f16_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                    float _cvt_f32_f16_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                    float _mul_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_20), "f"(rs_a_f));
                    float _mul_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_21), "f"(rs_a_f));
                    float2 scale0_822 = make_float2(_mul_20, _mul_20);
                    float2 scale1_823 = make_float2(_mul_21, _mul_21);
                    float _unpack_fp4_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_817[0]));
                        (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(a_817[0]));
                        #endif
                    }
                    float2 a_824 = make_float2(acc_821[0], acc_821[1]);
                    float2 q_825 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                    float2 r_826 = fma_f32x2_rn_noftz(q_825, scale0_822, a_824);
                    acc_821[0] = r_826.x;
                    acc_821[1] = r_826.y;
                    float2 a_827 = make_float2(acc_821[2], acc_821[3]);
                    float2 q_828 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                    float2 r_829 = fma_f32x2_rn_noftz(q_828, scale0_822, a_827);
                    acc_821[2] = r_829.x;
                    acc_821[3] = r_829.y;
                    float2 a_830 = make_float2(acc_821[4], acc_821[5]);
                    float2 q_831 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                    float2 r_832 = fma_f32x2_rn_noftz(q_831, scale0_822, a_830);
                    acc_821[4] = r_832.x;
                    acc_821[5] = r_832.y;
                    float2 a_833 = make_float2(acc_821[6], acc_821[7]);
                    float2 q_834 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                    float2 r_835 = fma_f32x2_rn_noftz(q_834, scale0_822, a_833);
                    acc_821[6] = r_835.x;
                    acc_821[7] = r_835.y;
                    float _unpack_fp4_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_817[1 + 0]));
                        (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(a_817[1 + 0]));
                        #endif
                    }
                    float2 a_836 = make_float2(acc_821[8], acc_821[9]);
                    float2 q_837 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                    float2 r_838 = fma_f32x2_rn_noftz(q_837, scale0_822, a_836);
                    acc_821[8] = r_838.x;
                    acc_821[9] = r_838.y;
                    float2 a_839 = make_float2(acc_821[10], acc_821[11]);
                    float2 q_840 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                    float2 r_841 = fma_f32x2_rn_noftz(q_840, scale0_822, a_839);
                    acc_821[10] = r_841.x;
                    acc_821[11] = r_841.y;
                    float2 a_842 = make_float2(acc_821[12], acc_821[13]);
                    float2 q_843 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                    float2 r_844 = fma_f32x2_rn_noftz(q_843, scale0_822, a_842);
                    acc_821[12] = r_844.x;
                    acc_821[13] = r_844.y;
                    float2 a_845 = make_float2(acc_821[14], acc_821[15]);
                    float2 q_846 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                    float2 r_847 = fma_f32x2_rn_noftz(q_846, scale0_822, a_845);
                    acc_821[14] = r_847.x;
                    acc_821[15] = r_847.y;
                    float _unpack_fp4_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_817[2 + 0]));
                        (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(a_817[2 + 0]));
                        #endif
                    }
                    float2 a_848 = make_float2(acc_821[16], acc_821[17]);
                    float2 q_849 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                    float2 r_850 = fma_f32x2_rn_noftz(q_849, scale1_823, a_848);
                    acc_821[16] = r_850.x;
                    acc_821[17] = r_850.y;
                    float2 a_851 = make_float2(acc_821[18], acc_821[19]);
                    float2 q_852 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                    float2 r_853 = fma_f32x2_rn_noftz(q_852, scale1_823, a_851);
                    acc_821[18] = r_853.x;
                    acc_821[19] = r_853.y;
                    float2 a_854 = make_float2(acc_821[20], acc_821[21]);
                    float2 q_855 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                    float2 r_856 = fma_f32x2_rn_noftz(q_855, scale1_823, a_854);
                    acc_821[20] = r_856.x;
                    acc_821[21] = r_856.y;
                    float2 a_857 = make_float2(acc_821[22], acc_821[23]);
                    float2 q_858 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                    float2 r_859 = fma_f32x2_rn_noftz(q_858, scale1_823, a_857);
                    acc_821[22] = r_859.x;
                    acc_821[23] = r_859.y;
                    float _unpack_fp4_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(a_817[3 + 0]));
                        (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(a_817[3 + 0]));
                        #endif
                    }
                    float2 a_860 = make_float2(acc_821[24], acc_821[25]);
                    float2 q_861 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                    float2 r_862 = fma_f32x2_rn_noftz(q_861, scale1_823, a_860);
                    acc_821[24] = r_862.x;
                    acc_821[25] = r_862.y;
                    float2 a_863 = make_float2(acc_821[26], acc_821[27]);
                    float2 q_864 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                    float2 r_865 = fma_f32x2_rn_noftz(q_864, scale1_823, a_863);
                    acc_821[26] = r_865.x;
                    acc_821[27] = r_865.y;
                    float2 a_866 = make_float2(acc_821[28], acc_821[29]);
                    float2 q_867 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                    float2 r_868 = fma_f32x2_rn_noftz(q_867, scale1_823, a_866);
                    acc_821[28] = r_868.x;
                    acc_821[29] = r_868.y;
                    float2 a_869 = make_float2(acc_821[30], acc_821[31]);
                    float2 q_870 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                    float2 r_871 = fma_f32x2_rn_noftz(q_870, scale1_823, a_869);
                    acc_821[30] = r_871.x;
                    acc_821[31] = r_871.y;
                    uint32_t _e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(b_sc_820)));
                    float _cvt_f32_f16_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                    float _cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                    float _mul_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_22), "f"(rs_b_f));
                    float _mul_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_23), "f"(rs_b_f));
                    float2 scale0_872 = make_float2(_mul_22, _mul_22);
                    float2 scale1_873 = make_float2(_mul_23, _mul_23);
                    float _unpack_fp4_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_818[0]));
                        (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(b_818[0]));
                        #endif
                    }
                    float2 a_874 = make_float2(acc_821[0], acc_821[1]);
                    float2 q_875 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                    float2 r_876 = fma_f32x2_rn_noftz(q_875, scale0_872, a_874);
                    acc_821[0] = r_876.x;
                    acc_821[1] = r_876.y;
                    float2 a_877 = make_float2(acc_821[2], acc_821[3]);
                    float2 q_878 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                    float2 r_879 = fma_f32x2_rn_noftz(q_878, scale0_872, a_877);
                    acc_821[2] = r_879.x;
                    acc_821[3] = r_879.y;
                    float2 a_880 = make_float2(acc_821[4], acc_821[5]);
                    float2 q_881 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                    float2 r_882 = fma_f32x2_rn_noftz(q_881, scale0_872, a_880);
                    acc_821[4] = r_882.x;
                    acc_821[5] = r_882.y;
                    float2 a_883 = make_float2(acc_821[6], acc_821[7]);
                    float2 q_884 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                    float2 r_885 = fma_f32x2_rn_noftz(q_884, scale0_872, a_883);
                    acc_821[6] = r_885.x;
                    acc_821[7] = r_885.y;
                    float _unpack_fp4_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_818[1 + 0]));
                        (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(b_818[1 + 0]));
                        #endif
                    }
                    float2 a_886 = make_float2(acc_821[8], acc_821[9]);
                    float2 q_887 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                    float2 r_888 = fma_f32x2_rn_noftz(q_887, scale0_872, a_886);
                    acc_821[8] = r_888.x;
                    acc_821[9] = r_888.y;
                    float2 a_889 = make_float2(acc_821[10], acc_821[11]);
                    float2 q_890 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                    float2 r_891 = fma_f32x2_rn_noftz(q_890, scale0_872, a_889);
                    acc_821[10] = r_891.x;
                    acc_821[11] = r_891.y;
                    float2 a_892 = make_float2(acc_821[12], acc_821[13]);
                    float2 q_893 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                    float2 r_894 = fma_f32x2_rn_noftz(q_893, scale0_872, a_892);
                    acc_821[12] = r_894.x;
                    acc_821[13] = r_894.y;
                    float2 a_895 = make_float2(acc_821[14], acc_821[15]);
                    float2 q_896 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                    float2 r_897 = fma_f32x2_rn_noftz(q_896, scale0_872, a_895);
                    acc_821[14] = r_897.x;
                    acc_821[15] = r_897.y;
                    float _unpack_fp4_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_818[2 + 0]));
                        (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(b_818[2 + 0]));
                        #endif
                    }
                    float2 a_898 = make_float2(acc_821[16], acc_821[17]);
                    float2 q_899 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                    float2 r_900 = fma_f32x2_rn_noftz(q_899, scale1_873, a_898);
                    acc_821[16] = r_900.x;
                    acc_821[17] = r_900.y;
                    float2 a_901 = make_float2(acc_821[18], acc_821[19]);
                    float2 q_902 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                    float2 r_903 = fma_f32x2_rn_noftz(q_902, scale1_873, a_901);
                    acc_821[18] = r_903.x;
                    acc_821[19] = r_903.y;
                    float2 a_904 = make_float2(acc_821[20], acc_821[21]);
                    float2 q_905 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                    float2 r_906 = fma_f32x2_rn_noftz(q_905, scale1_873, a_904);
                    acc_821[20] = r_906.x;
                    acc_821[21] = r_906.y;
                    float2 a_907 = make_float2(acc_821[22], acc_821[23]);
                    float2 q_908 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                    float2 r_909 = fma_f32x2_rn_noftz(q_908, scale1_873, a_907);
                    acc_821[22] = r_909.x;
                    acc_821[23] = r_909.y;
                    float _unpack_fp4_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(b_818[3 + 0]));
                        (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(b_818[3 + 0]));
                        #endif
                    }
                    float2 a_910 = make_float2(acc_821[24], acc_821[25]);
                    float2 q_911 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                    float2 r_912 = fma_f32x2_rn_noftz(q_911, scale1_873, a_910);
                    acc_821[24] = r_912.x;
                    acc_821[25] = r_912.y;
                    float2 a_913 = make_float2(acc_821[26], acc_821[27]);
                    float2 q_914 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                    float2 r_915 = fma_f32x2_rn_noftz(q_914, scale1_873, a_913);
                    acc_821[26] = r_915.x;
                    acc_821[27] = r_915.y;
                    float2 a_916 = make_float2(acc_821[28], acc_821[29]);
                    float2 q_917 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                    float2 r_918 = fma_f32x2_rn_noftz(q_917, scale1_873, a_916);
                    acc_821[28] = r_918.x;
                    acc_821[29] = r_918.y;
                    float2 a_919 = make_float2(acc_821[30], acc_821[31]);
                    float2 q_920 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                    float2 r_921 = fma_f32x2_rn_noftz(q_920, scale1_873, a_919);
                    acc_821[30] = r_921.x;
                    acc_821[31] = r_921.y;
                    unsigned int t_922[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_922[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_922[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_922[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_922[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_922[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 3))));
                    bool fodd_923 = (f & 1) != 0;
                    bool fhi_924 = (f & 2) != 0;
                    unsigned int s_925[16];
                    s_925[0] = t_922[0];
                    s_925[1] = t_922[1];
                    s_925[2] = t_922[2];
                    s_925[3] = t_922[3];
                    s_925[4] = t_922[4];
                    s_925[5] = t_922[5];
                    s_925[6] = t_922[6];
                    s_925[7] = t_922[7];
                    s_925[8] = t_922[8];
                    s_925[9] = t_922[9];
                    s_925[10] = t_922[10];
                    s_925[11] = t_922[11];
                    s_925[12] = t_922[12];
                    s_925[13] = t_922[13];
                    s_925[14] = t_922[14];
                    s_925[15] = t_922[15];
                    if (fodd_923) {
                        s_925[0] = t_922[4];
                        s_925[4] = t_922[0];
                        s_925[8] = t_922[12];
                        s_925[12] = t_922[8];
                        s_925[1] = t_922[5];
                        s_925[5] = t_922[1];
                        s_925[9] = t_922[13];
                        s_925[13] = t_922[9];
                        s_925[2] = t_922[6];
                        s_925[6] = t_922[2];
                        s_925[10] = t_922[14];
                        s_925[14] = t_922[10];
                        s_925[3] = t_922[7];
                        s_925[7] = t_922[3];
                        s_925[11] = t_922[15];
                        s_925[15] = t_922[11];
                    }
                    unsigned int r_926[16];
                    r_926[0] = s_925[0];
                    r_926[1] = s_925[1];
                    r_926[2] = s_925[2];
                    r_926[3] = s_925[3];
                    r_926[4] = s_925[4];
                    r_926[5] = s_925[5];
                    r_926[6] = s_925[6];
                    r_926[7] = s_925[7];
                    r_926[8] = s_925[8];
                    r_926[9] = s_925[9];
                    r_926[10] = s_925[10];
                    r_926[11] = s_925[11];
                    r_926[12] = s_925[12];
                    r_926[13] = s_925[13];
                    r_926[14] = s_925[14];
                    r_926[15] = s_925[15];
                    if (fhi_924) {
                        r_926[0] = s_925[8];
                        r_926[4] = s_925[12];
                        r_926[8] = s_925[0];
                        r_926[12] = s_925[4];
                        r_926[1] = s_925[9];
                        r_926[5] = s_925[13];
                        r_926[9] = s_925[1];
                        r_926[13] = s_925[5];
                        r_926[2] = s_925[10];
                        r_926[6] = s_925[14];
                        r_926[10] = s_925[2];
                        r_926[14] = s_925[6];
                        r_926[3] = s_925[11];
                        r_926[7] = s_925[15];
                        r_926[11] = s_925[3];
                        r_926[15] = s_925[7];
                    }
                    float2 a2_927 = make_float2(acc_821[0], acc_821[1]);
                    float2 z_928 = add_f32x2_noftz(a2_927, make_float2(__uint_as_float(r_926[0] << 16), __uint_as_float(r_926[0] & 4294901760u)));
                    acc_821[0] = z_928.x;
                    acc_821[1] = z_928.y;
                    float2 ss2_929 = fma_f32x2_rn_noftz(z_928, z_928, make_float2(ssx, ssy));
                    ssx = ss2_929.x;
                    ssy = ss2_929.y;
                    float2 a2_930 = make_float2(acc_821[2], acc_821[3]);
                    float2 z_931 = add_f32x2_noftz(a2_930, make_float2(__uint_as_float(r_926[1] << 16), __uint_as_float(r_926[1] & 4294901760u)));
                    acc_821[2] = z_931.x;
                    acc_821[3] = z_931.y;
                    float2 ss2_932 = fma_f32x2_rn_noftz(z_931, z_931, make_float2(ssx, ssy));
                    ssx = ss2_932.x;
                    ssy = ss2_932.y;
                    float2 a2_933 = make_float2(acc_821[4], acc_821[5]);
                    float2 z_934 = add_f32x2_noftz(a2_933, make_float2(__uint_as_float(r_926[2] << 16), __uint_as_float(r_926[2] & 4294901760u)));
                    acc_821[4] = z_934.x;
                    acc_821[5] = z_934.y;
                    float2 ss2_935 = fma_f32x2_rn_noftz(z_934, z_934, make_float2(ssx, ssy));
                    ssx = ss2_935.x;
                    ssy = ss2_935.y;
                    float2 a2_936 = make_float2(acc_821[6], acc_821[7]);
                    float2 z_937 = add_f32x2_noftz(a2_936, make_float2(__uint_as_float(r_926[3] << 16), __uint_as_float(r_926[3] & 4294901760u)));
                    acc_821[6] = z_937.x;
                    acc_821[7] = z_937.y;
                    float2 ss2_938 = fma_f32x2_rn_noftz(z_937, z_937, make_float2(ssx, ssy));
                    ssx = ss2_938.x;
                    ssy = ss2_938.y;
                    float2 a2_939 = make_float2(acc_821[8], acc_821[9]);
                    float2 z_940 = add_f32x2_noftz(a2_939, make_float2(__uint_as_float(r_926[4] << 16), __uint_as_float(r_926[4] & 4294901760u)));
                    acc_821[8] = z_940.x;
                    acc_821[9] = z_940.y;
                    float2 ss2_941 = fma_f32x2_rn_noftz(z_940, z_940, make_float2(ssx, ssy));
                    ssx = ss2_941.x;
                    ssy = ss2_941.y;
                    float2 a2_942 = make_float2(acc_821[10], acc_821[11]);
                    float2 z_943 = add_f32x2_noftz(a2_942, make_float2(__uint_as_float(r_926[5] << 16), __uint_as_float(r_926[5] & 4294901760u)));
                    acc_821[10] = z_943.x;
                    acc_821[11] = z_943.y;
                    float2 ss2_944 = fma_f32x2_rn_noftz(z_943, z_943, make_float2(ssx, ssy));
                    ssx = ss2_944.x;
                    ssy = ss2_944.y;
                    float2 a2_945 = make_float2(acc_821[12], acc_821[13]);
                    float2 z_946 = add_f32x2_noftz(a2_945, make_float2(__uint_as_float(r_926[6] << 16), __uint_as_float(r_926[6] & 4294901760u)));
                    acc_821[12] = z_946.x;
                    acc_821[13] = z_946.y;
                    float2 ss2_947 = fma_f32x2_rn_noftz(z_946, z_946, make_float2(ssx, ssy));
                    ssx = ss2_947.x;
                    ssy = ss2_947.y;
                    float2 a2_948 = make_float2(acc_821[14], acc_821[15]);
                    float2 z_949 = add_f32x2_noftz(a2_948, make_float2(__uint_as_float(r_926[7] << 16), __uint_as_float(r_926[7] & 4294901760u)));
                    acc_821[14] = z_949.x;
                    acc_821[15] = z_949.y;
                    float2 ss2_950 = fma_f32x2_rn_noftz(z_949, z_949, make_float2(ssx, ssy));
                    ssx = ss2_950.x;
                    ssy = ss2_950.y;
                    float2 a2_951 = make_float2(acc_821[16], acc_821[17]);
                    float2 z_952 = add_f32x2_noftz(a2_951, make_float2(__uint_as_float(r_926[8] << 16), __uint_as_float(r_926[8] & 4294901760u)));
                    acc_821[16] = z_952.x;
                    acc_821[17] = z_952.y;
                    float2 ss2_953 = fma_f32x2_rn_noftz(z_952, z_952, make_float2(ssx, ssy));
                    ssx = ss2_953.x;
                    ssy = ss2_953.y;
                    float2 a2_954 = make_float2(acc_821[18], acc_821[19]);
                    float2 z_955 = add_f32x2_noftz(a2_954, make_float2(__uint_as_float(r_926[9] << 16), __uint_as_float(r_926[9] & 4294901760u)));
                    acc_821[18] = z_955.x;
                    acc_821[19] = z_955.y;
                    float2 ss2_956 = fma_f32x2_rn_noftz(z_955, z_955, make_float2(ssx, ssy));
                    ssx = ss2_956.x;
                    ssy = ss2_956.y;
                    float2 a2_957 = make_float2(acc_821[20], acc_821[21]);
                    float2 z_958 = add_f32x2_noftz(a2_957, make_float2(__uint_as_float(r_926[10] << 16), __uint_as_float(r_926[10] & 4294901760u)));
                    acc_821[20] = z_958.x;
                    acc_821[21] = z_958.y;
                    float2 ss2_959 = fma_f32x2_rn_noftz(z_958, z_958, make_float2(ssx, ssy));
                    ssx = ss2_959.x;
                    ssy = ss2_959.y;
                    float2 a2_960 = make_float2(acc_821[22], acc_821[23]);
                    float2 z_961 = add_f32x2_noftz(a2_960, make_float2(__uint_as_float(r_926[11] << 16), __uint_as_float(r_926[11] & 4294901760u)));
                    acc_821[22] = z_961.x;
                    acc_821[23] = z_961.y;
                    float2 ss2_962 = fma_f32x2_rn_noftz(z_961, z_961, make_float2(ssx, ssy));
                    ssx = ss2_962.x;
                    ssy = ss2_962.y;
                    float2 a2_963 = make_float2(acc_821[24], acc_821[25]);
                    float2 z_964 = add_f32x2_noftz(a2_963, make_float2(__uint_as_float(r_926[12] << 16), __uint_as_float(r_926[12] & 4294901760u)));
                    acc_821[24] = z_964.x;
                    acc_821[25] = z_964.y;
                    float2 ss2_965 = fma_f32x2_rn_noftz(z_964, z_964, make_float2(ssx, ssy));
                    ssx = ss2_965.x;
                    ssy = ss2_965.y;
                    float2 a2_966 = make_float2(acc_821[26], acc_821[27]);
                    float2 z_967 = add_f32x2_noftz(a2_966, make_float2(__uint_as_float(r_926[13] << 16), __uint_as_float(r_926[13] & 4294901760u)));
                    acc_821[26] = z_967.x;
                    acc_821[27] = z_967.y;
                    float2 ss2_968 = fma_f32x2_rn_noftz(z_967, z_967, make_float2(ssx, ssy));
                    ssx = ss2_968.x;
                    ssy = ss2_968.y;
                    float2 a2_969 = make_float2(acc_821[28], acc_821[29]);
                    float2 z_970 = add_f32x2_noftz(a2_969, make_float2(__uint_as_float(r_926[14] << 16), __uint_as_float(r_926[14] & 4294901760u)));
                    acc_821[28] = z_970.x;
                    acc_821[29] = z_970.y;
                    float2 ss2_971 = fma_f32x2_rn_noftz(z_970, z_970, make_float2(ssx, ssy));
                    ssx = ss2_971.x;
                    ssy = ss2_971.y;
                    float2 a2_972 = make_float2(acc_821[30], acc_821[31]);
                    float2 z_973 = add_f32x2_noftz(a2_972, make_float2(__uint_as_float(r_926[15] << 16), __uint_as_float(r_926[15] & 4294901760u)));
                    acc_821[30] = z_973.x;
                    acc_821[31] = z_973.y;
                    float2 ss2_974 = fma_f32x2_rn_noftz(z_973, z_973, make_float2(ssx, ssy));
                    ssx = ss2_974.x;
                    ssy = ss2_974.y;
                    unsigned int acc_821_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_821[_lp*2 + 0], acc_821[_lp*2+1 + 0]));
                        acc_821_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    bool fodd_975 = (f & 1) != 0;
                    bool fhi_976 = (f & 2) != 0;
                    unsigned int s_977[16];
                    s_977[0] = acc_821_bf16[0];
                    s_977[1] = acc_821_bf16[1];
                    s_977[2] = acc_821_bf16[2];
                    s_977[3] = acc_821_bf16[3];
                    s_977[4] = acc_821_bf16[4];
                    s_977[5] = acc_821_bf16[5];
                    s_977[6] = acc_821_bf16[6];
                    s_977[7] = acc_821_bf16[7];
                    s_977[8] = acc_821_bf16[8];
                    s_977[9] = acc_821_bf16[9];
                    s_977[10] = acc_821_bf16[10];
                    s_977[11] = acc_821_bf16[11];
                    s_977[12] = acc_821_bf16[12];
                    s_977[13] = acc_821_bf16[13];
                    s_977[14] = acc_821_bf16[14];
                    s_977[15] = acc_821_bf16[15];
                    if (fodd_975) {
                        s_977[0] = acc_821_bf16[4];
                        s_977[4] = acc_821_bf16[0];
                        s_977[8] = acc_821_bf16[12];
                        s_977[12] = acc_821_bf16[8];
                        s_977[1] = acc_821_bf16[5];
                        s_977[5] = acc_821_bf16[1];
                        s_977[9] = acc_821_bf16[13];
                        s_977[13] = acc_821_bf16[9];
                        s_977[2] = acc_821_bf16[6];
                        s_977[6] = acc_821_bf16[2];
                        s_977[10] = acc_821_bf16[14];
                        s_977[14] = acc_821_bf16[10];
                        s_977[3] = acc_821_bf16[7];
                        s_977[7] = acc_821_bf16[3];
                        s_977[11] = acc_821_bf16[15];
                        s_977[15] = acc_821_bf16[11];
                    }
                    unsigned int r_978[16];
                    r_978[0] = s_977[0];
                    r_978[1] = s_977[1];
                    r_978[2] = s_977[2];
                    r_978[3] = s_977[3];
                    r_978[4] = s_977[4];
                    r_978[5] = s_977[5];
                    r_978[6] = s_977[6];
                    r_978[7] = s_977[7];
                    r_978[8] = s_977[8];
                    r_978[9] = s_977[9];
                    r_978[10] = s_977[10];
                    r_978[11] = s_977[11];
                    r_978[12] = s_977[12];
                    r_978[13] = s_977[13];
                    r_978[14] = s_977[14];
                    r_978[15] = s_977[15];
                    if (fhi_976) {
                        r_978[0] = s_977[8];
                        r_978[4] = s_977[12];
                        r_978[8] = s_977[0];
                        r_978[12] = s_977[4];
                        r_978[1] = s_977[9];
                        r_978[5] = s_977[13];
                        r_978[9] = s_977[1];
                        r_978[13] = s_977[5];
                        r_978[2] = s_977[10];
                        r_978[6] = s_977[14];
                        r_978[10] = s_977[2];
                        r_978[14] = s_977[6];
                        r_978[3] = s_977[11];
                        r_978[7] = s_977[15];
                        r_978[11] = s_977[3];
                        r_978[15] = s_977[7];
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 0))), "r"(r_978[0]), "r"(r_978[1]), "r"(r_978[2]), "r"(r_978[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 1))), "r"(r_978[4]), "r"(r_978[5]), "r"(r_978[6]), "r"(r_978[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 2))), "r"(r_978[8]), "r"(r_978[9]), "r"(r_978[10]), "r"(r_978[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_814 + 16 * (f ^ 3))), "r"(r_978[12]), "r"(r_978[13]), "r"(r_978[14]), "r"(r_978[15]) : "memory");
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
                    #pragma unroll 2
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
                        float2 _f32x2_mul_a_20 = make_float2(__uint_as_float(x4[0] << 16), __uint_as_float(x4[0] & 4294901760u));
                        float2 tz;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&r2));
                        float2 tw = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[0] << 16), __uint_as_float(w4[0] & 4294901760u)));
                        float2 t_0;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&tz), "l"(*(const unsigned long long*)&tw));
                        y[0] = t_0.x;
                        y[1] = t_0.y;
                        float2 _f32x2_mul_a_21 = make_float2(__uint_as_float(x4[1] << 16), __uint_as_float(x4[1] & 4294901760u));
                        float2 tz_1;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&r2));
                        float2 tw_2 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[1] << 16), __uint_as_float(w4[1] & 4294901760u)));
                        float2 t_3;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&tz_1), "l"(*(const unsigned long long*)&tw_2));
                        y[2] = t_3.x;
                        y[3] = t_3.y;
                        float2 _f32x2_mul_a_22 = make_float2(__uint_as_float(x4[2] << 16), __uint_as_float(x4[2] & 4294901760u));
                        float2 tz_4;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&r2));
                        float2 tw_5 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[2] << 16), __uint_as_float(w4[2] & 4294901760u)));
                        float2 t_6;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&tz_4), "l"(*(const unsigned long long*)&tw_5));
                        y[4] = t_6.x;
                        y[5] = t_6.y;
                        float2 _f32x2_mul_a_23 = make_float2(__uint_as_float(x4[3] << 16), __uint_as_float(x4[3] & 4294901760u));
                        float2 tz_7;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&r2));
                        float2 tw_8 = add_f32x2_noftz(b2, make_float2(__uint_as_float(w4[3] << 16), __uint_as_float(w4[3] & 4294901760u)));
                        float2 t_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&tz_7), "l"(*(const unsigned long long*)&tw_8));
                        y[6] = t_9.x;
                        y[7] = t_9.y;
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
            }
            if (lane_2 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
            if (tid == 0) {
                int base_1 = bid * 4;
                ctrl[base_1 + 2] = (unsigned int)rows;
                ctrl[base_1 + 3] = 1;
                ctrl[base_1] = call + 1;
            }
        }
    }
    // ---- Role: reset_warp ----
    if (warp == 8) {
        { // reset_warp_main
            int lane_0 = (int)lane;
            asm volatile("griddepcontrol.wait;" ::: "memory");
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
                long long lo = u * bid64_1 / g64_1;
                long long hi = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo + lane64;
                #pragma unroll 1
                for (long long g = g0; g < hi; g += 32) {
                    long long widx = v_base_1 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_1 = u_0 * bid64_1 / g64_1;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_1 + lane64;
                #pragma unroll 1
                for (long long g_1 = g0_3; g_1 < hi_2; g_1 += 32) {
                    long long widx_1 = s_base_1 + 16 * g_1 >> 2;
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
                for (long long g_2 = g0_7; g_2 < hi_6; g_2 += 32) {
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
