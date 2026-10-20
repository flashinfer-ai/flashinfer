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
#define SMEM_WSM_STAGE_BYTES 198016
#define SMEM_WSM_STRIDE 198016
#define SMEM_WSM32_OFF 13376
#define SMEM_WSM32_STAGE_BYTES 198016
#define SMEM_WSM32_STRIDE 198016
#define SMEM_TOTAL 211456
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

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
cake_fp4_allreduce_tp2_rmsnorm_nvfp4_nvfp4_pa_sop_pf(uint8_t* __restrict__ xq, float* __restrict__ gs, __nv_bfloat16* __restrict__ res_in, __nv_bfloat16* __restrict__ w, uint8_t* __restrict__ outq, uint8_t* __restrict__ out_sf, float* __restrict__ gs_out, __nv_bfloat16* __restrict__ res_out, uint8_t* __restrict__ os_wire, uint8_t* __restrict__ os_rs, long long* __restrict__ peer_ptrs, unsigned int* __restrict__ ctrl, int rank, int rows, int cap, float eps, float wbias)
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
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752 + 12464), res_in + (first_row * 6144), 12288, rb_full_addr + (warp_1) * 8);
                mbarrier_arrive_expect_tx(xb_full_addr + (warp_1) * 8, 3456);
                cp_async_bulk_gmem2smem(wsm_addr + (unsigned int)(warp_1 * 24752), xq + (first_row * 3456), 3456, xb_full_addr + (warp_1) * 8);
            }
            if (tid == 0) {
                ctl[0] = ctrl[0];
                ctl[1] = ctrl[2];
                ctl[2] = ctrl[3];
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
            #pragma unroll 1
            for (int k = 0; k < n + 1; k++) {
                if (n > k) {
                    long long row64 = (long long)(r0 + warp_1 + 8 * k);
                    int ob_off = xb_off + k % 3 * 3472;
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * lane_2)))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 32))))));
                    asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(reinterpret_cast<uint8_t*>(res_in) + (row64 * 12288 + (long long)(128 * (lane_2 + 64))))));
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
                    uint32_t _sysv_poll32_group_0[1];
                    do {
                        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(_sysv_poll32_group_0[0]) : "l"(reinterpret_cast<unsigned int*>(os_rs + m_row)) : "memory");
                    } while (
                        (_sysv_poll32_group_0[0] == 0x80000000u)
                    );
                    float rs_peer = __uint_as_float(_sysv_poll32_group_0[0]);
                    float rs_a = ((first) ? own_rs : rs_peer);
                    float rs_b = ((first) ? rs_peer : own_rs);
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
                    mbarrier_wait(rb_full_addr + (warp_1) * 8, kd & 1);
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
                    bool fhi = (f & 2) != 0;
                    bool fodd = (f & 1) != 0;
                    bool fhi_6 = (f & 2) != 0;
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
                    if (fhi_6) {
                        r[0] = s[2];
                        r[1] = s[3];
                        r[2] = s[0];
                        r[3] = s[1];
                    }
                    bool fodd_7 = (f & 1) != 0;
                    bool fhi_8 = (f & 2) != 0;
                    unsigned int s_9[4];
                    s_9[0] = b[0];
                    s_9[1] = b[1];
                    s_9[2] = b[2];
                    s_9[3] = b[3];
                    if (fodd_7) {
                        s_9[0] = b[1];
                        s_9[1] = b[0];
                        s_9[2] = b[3];
                        s_9[3] = b[2];
                    }
                    unsigned int r_10[4];
                    r_10[0] = s_9[0];
                    r_10[1] = s_9[1];
                    r_10[2] = s_9[2];
                    r_10[3] = s_9[3];
                    if (fhi_8) {
                        r_10[0] = s_9[2];
                        r_10[1] = s_9[3];
                        r_10[2] = s_9[0];
                        r_10[3] = s_9[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_0) : "h"((uint16_t)(a_sc)));
                    float _cvt_f32_f16_0;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(_e4m3x2_to_f16x2_0)));
                    float _cvt_f32_f16_1;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(_e4m3x2_to_f16x2_0 >> 16)));
                    float _mul_0;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_cvt_f32_f16_0), "f"(rs_a));
                    float _mul_1;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_cvt_f32_f16_1), "f"(rs_a));
                    float lo = _mul_0;
                    float hi = _mul_1;
                    if (fhi) {
                        lo = _mul_1;
                        hi = _mul_0;
                    }
                    float2 _f2_0 = make_float2(lo, lo);
                    float2 scale0 = _f2_0;
                    float2 _f2_1 = make_float2(hi, hi);
                    float2 scale1 = _f2_1;
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
                    float2 _f2_2 = make_float2(acc[0], acc[1]);
                    float2 a_11 = _f2_2;
                    float2 _f2_3 = make_float2(_unpack_fp4_0[0], _unpack_fp4_0[1]);
                    float2 q = _f2_3;
                    float2 r_12 = fma_f32x2_rn_noftz(q, scale0, a_11);
                    acc[0] = r_12.x;
                    acc[1] = r_12.y;
                    float2 _f2_4 = make_float2(acc[2], acc[3]);
                    float2 a_13 = _f2_4;
                    float2 _f2_5 = make_float2(_unpack_fp4_0[2], _unpack_fp4_0[3]);
                    float2 q_14 = _f2_5;
                    float2 r_15 = fma_f32x2_rn_noftz(q_14, scale0, a_13);
                    acc[2] = r_15.x;
                    acc[3] = r_15.y;
                    float2 _f2_6 = make_float2(acc[4], acc[5]);
                    float2 a_16 = _f2_6;
                    float2 _f2_7 = make_float2(_unpack_fp4_0[4], _unpack_fp4_0[5]);
                    float2 q_17 = _f2_7;
                    float2 r_18 = fma_f32x2_rn_noftz(q_17, scale0, a_16);
                    acc[4] = r_18.x;
                    acc[5] = r_18.y;
                    float2 _f2_8 = make_float2(acc[6], acc[7]);
                    float2 a_19 = _f2_8;
                    float2 _f2_9 = make_float2(_unpack_fp4_0[6], _unpack_fp4_0[7]);
                    float2 q_20 = _f2_9;
                    float2 r_21 = fma_f32x2_rn_noftz(q_20, scale0, a_19);
                    acc[6] = r_21.x;
                    acc[7] = r_21.y;
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
                    float2 _f2_10 = make_float2(acc[8], acc[9]);
                    float2 a_22 = _f2_10;
                    float2 _f2_11 = make_float2(_unpack_fp4_1[0], _unpack_fp4_1[1]);
                    float2 q_23 = _f2_11;
                    float2 r_24 = fma_f32x2_rn_noftz(q_23, scale0, a_22);
                    acc[8] = r_24.x;
                    acc[9] = r_24.y;
                    float2 _f2_12 = make_float2(acc[10], acc[11]);
                    float2 a_25 = _f2_12;
                    float2 _f2_13 = make_float2(_unpack_fp4_1[2], _unpack_fp4_1[3]);
                    float2 q_26 = _f2_13;
                    float2 r_27 = fma_f32x2_rn_noftz(q_26, scale0, a_25);
                    acc[10] = r_27.x;
                    acc[11] = r_27.y;
                    float2 _f2_14 = make_float2(acc[12], acc[13]);
                    float2 a_28 = _f2_14;
                    float2 _f2_15 = make_float2(_unpack_fp4_1[4], _unpack_fp4_1[5]);
                    float2 q_29 = _f2_15;
                    float2 r_30 = fma_f32x2_rn_noftz(q_29, scale0, a_28);
                    acc[12] = r_30.x;
                    acc[13] = r_30.y;
                    float2 _f2_16 = make_float2(acc[14], acc[15]);
                    float2 a_31 = _f2_16;
                    float2 _f2_17 = make_float2(_unpack_fp4_1[6], _unpack_fp4_1[7]);
                    float2 q_32 = _f2_17;
                    float2 r_33 = fma_f32x2_rn_noftz(q_32, scale0, a_31);
                    acc[14] = r_33.x;
                    acc[15] = r_33.y;
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
                    float2 _f2_18 = make_float2(acc[16], acc[17]);
                    float2 a_34 = _f2_18;
                    float2 _f2_19 = make_float2(_unpack_fp4_2[0], _unpack_fp4_2[1]);
                    float2 q_35 = _f2_19;
                    float2 r_36 = fma_f32x2_rn_noftz(q_35, scale1, a_34);
                    acc[16] = r_36.x;
                    acc[17] = r_36.y;
                    float2 _f2_20 = make_float2(acc[18], acc[19]);
                    float2 a_37 = _f2_20;
                    float2 _f2_21 = make_float2(_unpack_fp4_2[2], _unpack_fp4_2[3]);
                    float2 q_38 = _f2_21;
                    float2 r_39 = fma_f32x2_rn_noftz(q_38, scale1, a_37);
                    acc[18] = r_39.x;
                    acc[19] = r_39.y;
                    float2 _f2_22 = make_float2(acc[20], acc[21]);
                    float2 a_40 = _f2_22;
                    float2 _f2_23 = make_float2(_unpack_fp4_2[4], _unpack_fp4_2[5]);
                    float2 q_41 = _f2_23;
                    float2 r_42 = fma_f32x2_rn_noftz(q_41, scale1, a_40);
                    acc[20] = r_42.x;
                    acc[21] = r_42.y;
                    float2 _f2_24 = make_float2(acc[22], acc[23]);
                    float2 a_43 = _f2_24;
                    float2 _f2_25 = make_float2(_unpack_fp4_2[6], _unpack_fp4_2[7]);
                    float2 q_44 = _f2_25;
                    float2 r_45 = fma_f32x2_rn_noftz(q_44, scale1, a_43);
                    acc[22] = r_45.x;
                    acc[23] = r_45.y;
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
                    float2 _f2_26 = make_float2(acc[24], acc[25]);
                    float2 a_46 = _f2_26;
                    float2 _f2_27 = make_float2(_unpack_fp4_3[0], _unpack_fp4_3[1]);
                    float2 q_47 = _f2_27;
                    float2 r_48 = fma_f32x2_rn_noftz(q_47, scale1, a_46);
                    acc[24] = r_48.x;
                    acc[25] = r_48.y;
                    float2 _f2_28 = make_float2(acc[26], acc[27]);
                    float2 a_49 = _f2_28;
                    float2 _f2_29 = make_float2(_unpack_fp4_3[2], _unpack_fp4_3[3]);
                    float2 q_50 = _f2_29;
                    float2 r_51 = fma_f32x2_rn_noftz(q_50, scale1, a_49);
                    acc[26] = r_51.x;
                    acc[27] = r_51.y;
                    float2 _f2_30 = make_float2(acc[28], acc[29]);
                    float2 a_52 = _f2_30;
                    float2 _f2_31 = make_float2(_unpack_fp4_3[4], _unpack_fp4_3[5]);
                    float2 q_53 = _f2_31;
                    float2 r_54 = fma_f32x2_rn_noftz(q_53, scale1, a_52);
                    acc[28] = r_54.x;
                    acc[29] = r_54.y;
                    float2 _f2_32 = make_float2(acc[30], acc[31]);
                    float2 a_55 = _f2_32;
                    float2 _f2_33 = make_float2(_unpack_fp4_3[6], _unpack_fp4_3[7]);
                    float2 q_56 = _f2_33;
                    float2 r_57 = fma_f32x2_rn_noftz(q_56, scale1, a_55);
                    acc[30] = r_57.x;
                    acc[31] = r_57.y;
                    uint32_t _e4m3x2_to_f16x2_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_1) : "h"((uint16_t)(b_sc)));
                    float _cvt_f32_f16_2;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(_e4m3x2_to_f16x2_1)));
                    float _cvt_f32_f16_3;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(_e4m3x2_to_f16x2_1 >> 16)));
                    float _mul_2;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_cvt_f32_f16_2), "f"(rs_b));
                    float _mul_3;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_cvt_f32_f16_3), "f"(rs_b));
                    float lo_58 = _mul_2;
                    float hi_59 = _mul_3;
                    if (fhi) {
                        lo_58 = _mul_3;
                        hi_59 = _mul_2;
                    }
                    float2 _f2_34 = make_float2(lo_58, lo_58);
                    float2 scale0_60 = _f2_34;
                    float2 _f2_35 = make_float2(hi_59, hi_59);
                    float2 scale1_61 = _f2_35;
                    float _unpack_fp4_4[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_10[0]));
                        (&_unpack_fp4_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_4[0]), "=f"(_unpack_fp4_4[1]), "=f"(_unpack_fp4_4[2]), "=f"(_unpack_fp4_4[3]), "=f"(_unpack_fp4_4[4]), "=f"(_unpack_fp4_4[5]), "=f"(_unpack_fp4_4[6]), "=f"(_unpack_fp4_4[7]) : "r"(r_10[0]));
                        #endif
                    }
                    float2 _f2_36 = make_float2(acc[0], acc[1]);
                    float2 a_62 = _f2_36;
                    float2 _f2_37 = make_float2(_unpack_fp4_4[0], _unpack_fp4_4[1]);
                    float2 q_63 = _f2_37;
                    float2 r_64 = fma_f32x2_rn_noftz(q_63, scale0_60, a_62);
                    acc[0] = r_64.x;
                    acc[1] = r_64.y;
                    float2 _f2_38 = make_float2(acc[2], acc[3]);
                    float2 a_65 = _f2_38;
                    float2 _f2_39 = make_float2(_unpack_fp4_4[2], _unpack_fp4_4[3]);
                    float2 q_66 = _f2_39;
                    float2 r_67 = fma_f32x2_rn_noftz(q_66, scale0_60, a_65);
                    acc[2] = r_67.x;
                    acc[3] = r_67.y;
                    float2 _f2_40 = make_float2(acc[4], acc[5]);
                    float2 a_68 = _f2_40;
                    float2 _f2_41 = make_float2(_unpack_fp4_4[4], _unpack_fp4_4[5]);
                    float2 q_69 = _f2_41;
                    float2 r_70 = fma_f32x2_rn_noftz(q_69, scale0_60, a_68);
                    acc[4] = r_70.x;
                    acc[5] = r_70.y;
                    float2 _f2_42 = make_float2(acc[6], acc[7]);
                    float2 a_71 = _f2_42;
                    float2 _f2_43 = make_float2(_unpack_fp4_4[6], _unpack_fp4_4[7]);
                    float2 q_72 = _f2_43;
                    float2 r_73 = fma_f32x2_rn_noftz(q_72, scale0_60, a_71);
                    acc[6] = r_73.x;
                    acc[7] = r_73.y;
                    float _unpack_fp4_5[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_10[1 + 0]));
                        (&_unpack_fp4_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_5[0]), "=f"(_unpack_fp4_5[1]), "=f"(_unpack_fp4_5[2]), "=f"(_unpack_fp4_5[3]), "=f"(_unpack_fp4_5[4]), "=f"(_unpack_fp4_5[5]), "=f"(_unpack_fp4_5[6]), "=f"(_unpack_fp4_5[7]) : "r"(r_10[1 + 0]));
                        #endif
                    }
                    float2 _f2_44 = make_float2(acc[8], acc[9]);
                    float2 a_74 = _f2_44;
                    float2 _f2_45 = make_float2(_unpack_fp4_5[0], _unpack_fp4_5[1]);
                    float2 q_75 = _f2_45;
                    float2 r_76 = fma_f32x2_rn_noftz(q_75, scale0_60, a_74);
                    acc[8] = r_76.x;
                    acc[9] = r_76.y;
                    float2 _f2_46 = make_float2(acc[10], acc[11]);
                    float2 a_77 = _f2_46;
                    float2 _f2_47 = make_float2(_unpack_fp4_5[2], _unpack_fp4_5[3]);
                    float2 q_78 = _f2_47;
                    float2 r_79 = fma_f32x2_rn_noftz(q_78, scale0_60, a_77);
                    acc[10] = r_79.x;
                    acc[11] = r_79.y;
                    float2 _f2_48 = make_float2(acc[12], acc[13]);
                    float2 a_80 = _f2_48;
                    float2 _f2_49 = make_float2(_unpack_fp4_5[4], _unpack_fp4_5[5]);
                    float2 q_81 = _f2_49;
                    float2 r_82 = fma_f32x2_rn_noftz(q_81, scale0_60, a_80);
                    acc[12] = r_82.x;
                    acc[13] = r_82.y;
                    float2 _f2_50 = make_float2(acc[14], acc[15]);
                    float2 a_83 = _f2_50;
                    float2 _f2_51 = make_float2(_unpack_fp4_5[6], _unpack_fp4_5[7]);
                    float2 q_84 = _f2_51;
                    float2 r_85 = fma_f32x2_rn_noftz(q_84, scale0_60, a_83);
                    acc[14] = r_85.x;
                    acc[15] = r_85.y;
                    float _unpack_fp4_6[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_10[2 + 0]));
                        (&_unpack_fp4_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_6[0]), "=f"(_unpack_fp4_6[1]), "=f"(_unpack_fp4_6[2]), "=f"(_unpack_fp4_6[3]), "=f"(_unpack_fp4_6[4]), "=f"(_unpack_fp4_6[5]), "=f"(_unpack_fp4_6[6]), "=f"(_unpack_fp4_6[7]) : "r"(r_10[2 + 0]));
                        #endif
                    }
                    float2 _f2_52 = make_float2(acc[16], acc[17]);
                    float2 a_86 = _f2_52;
                    float2 _f2_53 = make_float2(_unpack_fp4_6[0], _unpack_fp4_6[1]);
                    float2 q_87 = _f2_53;
                    float2 r_88 = fma_f32x2_rn_noftz(q_87, scale1_61, a_86);
                    acc[16] = r_88.x;
                    acc[17] = r_88.y;
                    float2 _f2_54 = make_float2(acc[18], acc[19]);
                    float2 a_89 = _f2_54;
                    float2 _f2_55 = make_float2(_unpack_fp4_6[2], _unpack_fp4_6[3]);
                    float2 q_90 = _f2_55;
                    float2 r_91 = fma_f32x2_rn_noftz(q_90, scale1_61, a_89);
                    acc[18] = r_91.x;
                    acc[19] = r_91.y;
                    float2 _f2_56 = make_float2(acc[20], acc[21]);
                    float2 a_92 = _f2_56;
                    float2 _f2_57 = make_float2(_unpack_fp4_6[4], _unpack_fp4_6[5]);
                    float2 q_93 = _f2_57;
                    float2 r_94 = fma_f32x2_rn_noftz(q_93, scale1_61, a_92);
                    acc[20] = r_94.x;
                    acc[21] = r_94.y;
                    float2 _f2_58 = make_float2(acc[22], acc[23]);
                    float2 a_95 = _f2_58;
                    float2 _f2_59 = make_float2(_unpack_fp4_6[6], _unpack_fp4_6[7]);
                    float2 q_96 = _f2_59;
                    float2 r_97 = fma_f32x2_rn_noftz(q_96, scale1_61, a_95);
                    acc[22] = r_97.x;
                    acc[23] = r_97.y;
                    float _unpack_fp4_7[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_10[3 + 0]));
                        (&_unpack_fp4_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_7[0]), "=f"(_unpack_fp4_7[1]), "=f"(_unpack_fp4_7[2]), "=f"(_unpack_fp4_7[3]), "=f"(_unpack_fp4_7[4]), "=f"(_unpack_fp4_7[5]), "=f"(_unpack_fp4_7[6]), "=f"(_unpack_fp4_7[7]) : "r"(r_10[3 + 0]));
                        #endif
                    }
                    float2 _f2_60 = make_float2(acc[24], acc[25]);
                    float2 a_98 = _f2_60;
                    float2 _f2_61 = make_float2(_unpack_fp4_7[0], _unpack_fp4_7[1]);
                    float2 q_99 = _f2_61;
                    float2 r_100 = fma_f32x2_rn_noftz(q_99, scale1_61, a_98);
                    acc[24] = r_100.x;
                    acc[25] = r_100.y;
                    float2 _f2_62 = make_float2(acc[26], acc[27]);
                    float2 a_101 = _f2_62;
                    float2 _f2_63 = make_float2(_unpack_fp4_7[2], _unpack_fp4_7[3]);
                    float2 q_102 = _f2_63;
                    float2 r_103 = fma_f32x2_rn_noftz(q_102, scale1_61, a_101);
                    acc[26] = r_103.x;
                    acc[27] = r_103.y;
                    float2 _f2_64 = make_float2(acc[28], acc[29]);
                    float2 a_104 = _f2_64;
                    float2 _f2_65 = make_float2(_unpack_fp4_7[4], _unpack_fp4_7[5]);
                    float2 q_105 = _f2_65;
                    float2 r_106 = fma_f32x2_rn_noftz(q_105, scale1_61, a_104);
                    acc[28] = r_106.x;
                    acc[29] = r_106.y;
                    float2 _f2_66 = make_float2(acc[30], acc[31]);
                    float2 a_107 = _f2_66;
                    float2 _f2_67 = make_float2(_unpack_fp4_7[6], _unpack_fp4_7[7]);
                    float2 q_108 = _f2_67;
                    float2 r_109 = fma_f32x2_rn_noftz(q_108, scale1_61, a_107);
                    acc[30] = r_109.x;
                    acc[31] = r_109.y;
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
                    float2 _f2_68 = make_float2(acc[0], acc[1]);
                    float2 a2 = _f2_68;
                    float2 _f2_69 = make_float2(__uint_as_float(t[0] << 16), __uint_as_float(t[0] & 4294901760u));
                    float2 z = add_f32x2_noftz(a2, _f2_69);
                    acc[0] = z.x;
                    acc[1] = z.y;
                    float2 _f2_70 = make_float2(ssx, ssy);
                    float2 ss2 = fma_f32x2_rn_noftz(z, z, _f2_70);
                    ssx = ss2.x;
                    ssy = ss2.y;
                    float2 _f2_71 = make_float2(acc[2], acc[3]);
                    float2 a2_110 = _f2_71;
                    float2 _f2_72 = make_float2(__uint_as_float(t[1] << 16), __uint_as_float(t[1] & 4294901760u));
                    float2 z_111 = add_f32x2_noftz(a2_110, _f2_72);
                    acc[2] = z_111.x;
                    acc[3] = z_111.y;
                    float2 _f2_73 = make_float2(ssx, ssy);
                    float2 ss2_112 = fma_f32x2_rn_noftz(z_111, z_111, _f2_73);
                    ssx = ss2_112.x;
                    ssy = ss2_112.y;
                    float2 _f2_74 = make_float2(acc[4], acc[5]);
                    float2 a2_113 = _f2_74;
                    float2 _f2_75 = make_float2(__uint_as_float(t[2] << 16), __uint_as_float(t[2] & 4294901760u));
                    float2 z_114 = add_f32x2_noftz(a2_113, _f2_75);
                    acc[4] = z_114.x;
                    acc[5] = z_114.y;
                    float2 _f2_76 = make_float2(ssx, ssy);
                    float2 ss2_115 = fma_f32x2_rn_noftz(z_114, z_114, _f2_76);
                    ssx = ss2_115.x;
                    ssy = ss2_115.y;
                    float2 _f2_77 = make_float2(acc[6], acc[7]);
                    float2 a2_116 = _f2_77;
                    float2 _f2_78 = make_float2(__uint_as_float(t[3] << 16), __uint_as_float(t[3] & 4294901760u));
                    float2 z_117 = add_f32x2_noftz(a2_116, _f2_78);
                    acc[6] = z_117.x;
                    acc[7] = z_117.y;
                    float2 _f2_79 = make_float2(ssx, ssy);
                    float2 ss2_118 = fma_f32x2_rn_noftz(z_117, z_117, _f2_79);
                    ssx = ss2_118.x;
                    ssy = ss2_118.y;
                    float2 _f2_80 = make_float2(acc[8], acc[9]);
                    float2 a2_119 = _f2_80;
                    float2 _f2_81 = make_float2(__uint_as_float(t[4] << 16), __uint_as_float(t[4] & 4294901760u));
                    float2 z_120 = add_f32x2_noftz(a2_119, _f2_81);
                    acc[8] = z_120.x;
                    acc[9] = z_120.y;
                    float2 _f2_82 = make_float2(ssx, ssy);
                    float2 ss2_121 = fma_f32x2_rn_noftz(z_120, z_120, _f2_82);
                    ssx = ss2_121.x;
                    ssy = ss2_121.y;
                    float2 _f2_83 = make_float2(acc[10], acc[11]);
                    float2 a2_122 = _f2_83;
                    float2 _f2_84 = make_float2(__uint_as_float(t[5] << 16), __uint_as_float(t[5] & 4294901760u));
                    float2 z_123 = add_f32x2_noftz(a2_122, _f2_84);
                    acc[10] = z_123.x;
                    acc[11] = z_123.y;
                    float2 _f2_85 = make_float2(ssx, ssy);
                    float2 ss2_124 = fma_f32x2_rn_noftz(z_123, z_123, _f2_85);
                    ssx = ss2_124.x;
                    ssy = ss2_124.y;
                    float2 _f2_86 = make_float2(acc[12], acc[13]);
                    float2 a2_125 = _f2_86;
                    float2 _f2_87 = make_float2(__uint_as_float(t[6] << 16), __uint_as_float(t[6] & 4294901760u));
                    float2 z_126 = add_f32x2_noftz(a2_125, _f2_87);
                    acc[12] = z_126.x;
                    acc[13] = z_126.y;
                    float2 _f2_88 = make_float2(ssx, ssy);
                    float2 ss2_127 = fma_f32x2_rn_noftz(z_126, z_126, _f2_88);
                    ssx = ss2_127.x;
                    ssy = ss2_127.y;
                    float2 _f2_89 = make_float2(acc[14], acc[15]);
                    float2 a2_128 = _f2_89;
                    float2 _f2_90 = make_float2(__uint_as_float(t[7] << 16), __uint_as_float(t[7] & 4294901760u));
                    float2 z_129 = add_f32x2_noftz(a2_128, _f2_90);
                    acc[14] = z_129.x;
                    acc[15] = z_129.y;
                    float2 _f2_91 = make_float2(ssx, ssy);
                    float2 ss2_130 = fma_f32x2_rn_noftz(z_129, z_129, _f2_91);
                    ssx = ss2_130.x;
                    ssy = ss2_130.y;
                    float2 _f2_92 = make_float2(acc[16], acc[17]);
                    float2 a2_131 = _f2_92;
                    float2 _f2_93 = make_float2(__uint_as_float(t[8] << 16), __uint_as_float(t[8] & 4294901760u));
                    float2 z_132 = add_f32x2_noftz(a2_131, _f2_93);
                    acc[16] = z_132.x;
                    acc[17] = z_132.y;
                    float2 _f2_94 = make_float2(ssx, ssy);
                    float2 ss2_133 = fma_f32x2_rn_noftz(z_132, z_132, _f2_94);
                    ssx = ss2_133.x;
                    ssy = ss2_133.y;
                    float2 _f2_95 = make_float2(acc[18], acc[19]);
                    float2 a2_134 = _f2_95;
                    float2 _f2_96 = make_float2(__uint_as_float(t[9] << 16), __uint_as_float(t[9] & 4294901760u));
                    float2 z_135 = add_f32x2_noftz(a2_134, _f2_96);
                    acc[18] = z_135.x;
                    acc[19] = z_135.y;
                    float2 _f2_97 = make_float2(ssx, ssy);
                    float2 ss2_136 = fma_f32x2_rn_noftz(z_135, z_135, _f2_97);
                    ssx = ss2_136.x;
                    ssy = ss2_136.y;
                    float2 _f2_98 = make_float2(acc[20], acc[21]);
                    float2 a2_137 = _f2_98;
                    float2 _f2_99 = make_float2(__uint_as_float(t[10] << 16), __uint_as_float(t[10] & 4294901760u));
                    float2 z_138 = add_f32x2_noftz(a2_137, _f2_99);
                    acc[20] = z_138.x;
                    acc[21] = z_138.y;
                    float2 _f2_100 = make_float2(ssx, ssy);
                    float2 ss2_139 = fma_f32x2_rn_noftz(z_138, z_138, _f2_100);
                    ssx = ss2_139.x;
                    ssy = ss2_139.y;
                    float2 _f2_101 = make_float2(acc[22], acc[23]);
                    float2 a2_140 = _f2_101;
                    float2 _f2_102 = make_float2(__uint_as_float(t[11] << 16), __uint_as_float(t[11] & 4294901760u));
                    float2 z_141 = add_f32x2_noftz(a2_140, _f2_102);
                    acc[22] = z_141.x;
                    acc[23] = z_141.y;
                    float2 _f2_103 = make_float2(ssx, ssy);
                    float2 ss2_142 = fma_f32x2_rn_noftz(z_141, z_141, _f2_103);
                    ssx = ss2_142.x;
                    ssy = ss2_142.y;
                    float2 _f2_104 = make_float2(acc[24], acc[25]);
                    float2 a2_143 = _f2_104;
                    float2 _f2_105 = make_float2(__uint_as_float(t[12] << 16), __uint_as_float(t[12] & 4294901760u));
                    float2 z_144 = add_f32x2_noftz(a2_143, _f2_105);
                    acc[24] = z_144.x;
                    acc[25] = z_144.y;
                    float2 _f2_106 = make_float2(ssx, ssy);
                    float2 ss2_145 = fma_f32x2_rn_noftz(z_144, z_144, _f2_106);
                    ssx = ss2_145.x;
                    ssy = ss2_145.y;
                    float2 _f2_107 = make_float2(acc[26], acc[27]);
                    float2 a2_146 = _f2_107;
                    float2 _f2_108 = make_float2(__uint_as_float(t[13] << 16), __uint_as_float(t[13] & 4294901760u));
                    float2 z_147 = add_f32x2_noftz(a2_146, _f2_108);
                    acc[26] = z_147.x;
                    acc[27] = z_147.y;
                    float2 _f2_109 = make_float2(ssx, ssy);
                    float2 ss2_148 = fma_f32x2_rn_noftz(z_147, z_147, _f2_109);
                    ssx = ss2_148.x;
                    ssy = ss2_148.y;
                    float2 _f2_110 = make_float2(acc[28], acc[29]);
                    float2 a2_149 = _f2_110;
                    float2 _f2_111 = make_float2(__uint_as_float(t[14] << 16), __uint_as_float(t[14] & 4294901760u));
                    float2 z_150 = add_f32x2_noftz(a2_149, _f2_111);
                    acc[28] = z_150.x;
                    acc[29] = z_150.y;
                    float2 _f2_112 = make_float2(ssx, ssy);
                    float2 ss2_151 = fma_f32x2_rn_noftz(z_150, z_150, _f2_112);
                    ssx = ss2_151.x;
                    ssy = ss2_151.y;
                    float2 _f2_113 = make_float2(acc[30], acc[31]);
                    float2 a2_152 = _f2_113;
                    float2 _f2_114 = make_float2(__uint_as_float(t[15] << 16), __uint_as_float(t[15] & 4294901760u));
                    float2 z_153 = add_f32x2_noftz(a2_152, _f2_114);
                    acc[30] = z_153.x;
                    acc[31] = z_153.y;
                    float2 _f2_115 = make_float2(ssx, ssy);
                    float2 ss2_154 = fma_f32x2_rn_noftz(z_153, z_153, _f2_115);
                    ssx = ss2_154.x;
                    ssy = ss2_154.y;
                    unsigned int acc_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 0))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 1))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 2))), "r"(acc_bf16[8]), "r"(acc_bf16[9]), "r"(acc_bf16[10]), "r"(acc_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_5 + 16 * (f ^ 3))), "r"(acc_bf16[12]), "r"(acc_bf16[13]), "r"(acc_bf16[14]), "r"(acc_bf16[15]) : "memory");
                    int pp_155 = lane_2 + 32;
                    unsigned int own_156[4];
                    unsigned int own_sc_157[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_156[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_156[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_156[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_156[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_155)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_157[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_155)));
                    unsigned int a_158[4];
                    unsigned int b_159[4];
                    a_158[0] = ((first) ? own_156[0] : pv[4]);
                    b_159[0] = ((first) ? pv[4] : own_156[0]);
                    a_158[1] = ((first) ? own_156[1] : pv[5]);
                    b_159[1] = ((first) ? pv[5] : own_156[1]);
                    a_158[2] = ((first) ? own_156[2] : pv[6]);
                    b_159[2] = ((first) ? pv[6] : own_156[2]);
                    a_158[3] = ((first) ? own_156[3] : pv[7]);
                    b_159[3] = ((first) ? pv[7] : own_156[3]);
                    unsigned int a_sc_160 = ((first) ? own_sc_157[0] : ps[1]);
                    unsigned int b_sc_161 = ((first) ? ps[1] : own_sc_157[0]);
                    float acc_162[32] = {0};
                    bool fhi_163 = (f & 2) != 0;
                    bool fodd_164 = (f & 1) != 0;
                    bool fhi_165 = (f & 2) != 0;
                    unsigned int s_166[4];
                    s_166[0] = a_158[0];
                    s_166[1] = a_158[1];
                    s_166[2] = a_158[2];
                    s_166[3] = a_158[3];
                    if (fodd_164) {
                        s_166[0] = a_158[1];
                        s_166[1] = a_158[0];
                        s_166[2] = a_158[3];
                        s_166[3] = a_158[2];
                    }
                    unsigned int r_167[4];
                    r_167[0] = s_166[0];
                    r_167[1] = s_166[1];
                    r_167[2] = s_166[2];
                    r_167[3] = s_166[3];
                    if (fhi_165) {
                        r_167[0] = s_166[2];
                        r_167[1] = s_166[3];
                        r_167[2] = s_166[0];
                        r_167[3] = s_166[1];
                    }
                    bool fodd_168 = (f & 1) != 0;
                    bool fhi_169 = (f & 2) != 0;
                    unsigned int s_170[4];
                    s_170[0] = b_159[0];
                    s_170[1] = b_159[1];
                    s_170[2] = b_159[2];
                    s_170[3] = b_159[3];
                    if (fodd_168) {
                        s_170[0] = b_159[1];
                        s_170[1] = b_159[0];
                        s_170[2] = b_159[3];
                        s_170[3] = b_159[2];
                    }
                    unsigned int r_171[4];
                    r_171[0] = s_170[0];
                    r_171[1] = s_170[1];
                    r_171[2] = s_170[2];
                    r_171[3] = s_170[3];
                    if (fhi_169) {
                        r_171[0] = s_170[2];
                        r_171[1] = s_170[3];
                        r_171[2] = s_170[0];
                        r_171[3] = s_170[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_2) : "h"((uint16_t)(a_sc_160)));
                    float _cvt_f32_f16_4;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(_e4m3x2_to_f16x2_2)));
                    float _cvt_f32_f16_5;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(_e4m3x2_to_f16x2_2 >> 16)));
                    float _mul_4;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_cvt_f32_f16_4), "f"(rs_a));
                    float _mul_5;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_cvt_f32_f16_5), "f"(rs_a));
                    float lo_172 = _mul_4;
                    float hi_173 = _mul_5;
                    if (fhi_163) {
                        lo_172 = _mul_5;
                        hi_173 = _mul_4;
                    }
                    float2 _f2_116 = make_float2(lo_172, lo_172);
                    float2 scale0_174 = _f2_116;
                    float2 _f2_117 = make_float2(hi_173, hi_173);
                    float2 scale1_175 = _f2_117;
                    float _unpack_fp4_8[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_167[0]));
                        (&_unpack_fp4_8[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_8[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_8[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_8[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_8[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_8[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_8[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_8[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_8[0]), "=f"(_unpack_fp4_8[1]), "=f"(_unpack_fp4_8[2]), "=f"(_unpack_fp4_8[3]), "=f"(_unpack_fp4_8[4]), "=f"(_unpack_fp4_8[5]), "=f"(_unpack_fp4_8[6]), "=f"(_unpack_fp4_8[7]) : "r"(r_167[0]));
                        #endif
                    }
                    float2 _f2_118 = make_float2(acc_162[0], acc_162[1]);
                    float2 a_176 = _f2_118;
                    float2 _f2_119 = make_float2(_unpack_fp4_8[0], _unpack_fp4_8[1]);
                    float2 q_177 = _f2_119;
                    float2 r_178 = fma_f32x2_rn_noftz(q_177, scale0_174, a_176);
                    acc_162[0] = r_178.x;
                    acc_162[1] = r_178.y;
                    float2 _f2_120 = make_float2(acc_162[2], acc_162[3]);
                    float2 a_179 = _f2_120;
                    float2 _f2_121 = make_float2(_unpack_fp4_8[2], _unpack_fp4_8[3]);
                    float2 q_180 = _f2_121;
                    float2 r_181 = fma_f32x2_rn_noftz(q_180, scale0_174, a_179);
                    acc_162[2] = r_181.x;
                    acc_162[3] = r_181.y;
                    float2 _f2_122 = make_float2(acc_162[4], acc_162[5]);
                    float2 a_182 = _f2_122;
                    float2 _f2_123 = make_float2(_unpack_fp4_8[4], _unpack_fp4_8[5]);
                    float2 q_183 = _f2_123;
                    float2 r_184 = fma_f32x2_rn_noftz(q_183, scale0_174, a_182);
                    acc_162[4] = r_184.x;
                    acc_162[5] = r_184.y;
                    float2 _f2_124 = make_float2(acc_162[6], acc_162[7]);
                    float2 a_185 = _f2_124;
                    float2 _f2_125 = make_float2(_unpack_fp4_8[6], _unpack_fp4_8[7]);
                    float2 q_186 = _f2_125;
                    float2 r_187 = fma_f32x2_rn_noftz(q_186, scale0_174, a_185);
                    acc_162[6] = r_187.x;
                    acc_162[7] = r_187.y;
                    float _unpack_fp4_9[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_167[1 + 0]));
                        (&_unpack_fp4_9[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_9[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_9[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_9[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_9[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_9[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_9[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_9[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_9[0]), "=f"(_unpack_fp4_9[1]), "=f"(_unpack_fp4_9[2]), "=f"(_unpack_fp4_9[3]), "=f"(_unpack_fp4_9[4]), "=f"(_unpack_fp4_9[5]), "=f"(_unpack_fp4_9[6]), "=f"(_unpack_fp4_9[7]) : "r"(r_167[1 + 0]));
                        #endif
                    }
                    float2 _f2_126 = make_float2(acc_162[8], acc_162[9]);
                    float2 a_188 = _f2_126;
                    float2 _f2_127 = make_float2(_unpack_fp4_9[0], _unpack_fp4_9[1]);
                    float2 q_189 = _f2_127;
                    float2 r_190 = fma_f32x2_rn_noftz(q_189, scale0_174, a_188);
                    acc_162[8] = r_190.x;
                    acc_162[9] = r_190.y;
                    float2 _f2_128 = make_float2(acc_162[10], acc_162[11]);
                    float2 a_191 = _f2_128;
                    float2 _f2_129 = make_float2(_unpack_fp4_9[2], _unpack_fp4_9[3]);
                    float2 q_192 = _f2_129;
                    float2 r_193 = fma_f32x2_rn_noftz(q_192, scale0_174, a_191);
                    acc_162[10] = r_193.x;
                    acc_162[11] = r_193.y;
                    float2 _f2_130 = make_float2(acc_162[12], acc_162[13]);
                    float2 a_194 = _f2_130;
                    float2 _f2_131 = make_float2(_unpack_fp4_9[4], _unpack_fp4_9[5]);
                    float2 q_195 = _f2_131;
                    float2 r_196 = fma_f32x2_rn_noftz(q_195, scale0_174, a_194);
                    acc_162[12] = r_196.x;
                    acc_162[13] = r_196.y;
                    float2 _f2_132 = make_float2(acc_162[14], acc_162[15]);
                    float2 a_197 = _f2_132;
                    float2 _f2_133 = make_float2(_unpack_fp4_9[6], _unpack_fp4_9[7]);
                    float2 q_198 = _f2_133;
                    float2 r_199 = fma_f32x2_rn_noftz(q_198, scale0_174, a_197);
                    acc_162[14] = r_199.x;
                    acc_162[15] = r_199.y;
                    float _unpack_fp4_10[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_167[2 + 0]));
                        (&_unpack_fp4_10[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_10[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_10[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_10[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_10[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_10[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_10[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_10[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_10[0]), "=f"(_unpack_fp4_10[1]), "=f"(_unpack_fp4_10[2]), "=f"(_unpack_fp4_10[3]), "=f"(_unpack_fp4_10[4]), "=f"(_unpack_fp4_10[5]), "=f"(_unpack_fp4_10[6]), "=f"(_unpack_fp4_10[7]) : "r"(r_167[2 + 0]));
                        #endif
                    }
                    float2 _f2_134 = make_float2(acc_162[16], acc_162[17]);
                    float2 a_200 = _f2_134;
                    float2 _f2_135 = make_float2(_unpack_fp4_10[0], _unpack_fp4_10[1]);
                    float2 q_201 = _f2_135;
                    float2 r_202 = fma_f32x2_rn_noftz(q_201, scale1_175, a_200);
                    acc_162[16] = r_202.x;
                    acc_162[17] = r_202.y;
                    float2 _f2_136 = make_float2(acc_162[18], acc_162[19]);
                    float2 a_203 = _f2_136;
                    float2 _f2_137 = make_float2(_unpack_fp4_10[2], _unpack_fp4_10[3]);
                    float2 q_204 = _f2_137;
                    float2 r_205 = fma_f32x2_rn_noftz(q_204, scale1_175, a_203);
                    acc_162[18] = r_205.x;
                    acc_162[19] = r_205.y;
                    float2 _f2_138 = make_float2(acc_162[20], acc_162[21]);
                    float2 a_206 = _f2_138;
                    float2 _f2_139 = make_float2(_unpack_fp4_10[4], _unpack_fp4_10[5]);
                    float2 q_207 = _f2_139;
                    float2 r_208 = fma_f32x2_rn_noftz(q_207, scale1_175, a_206);
                    acc_162[20] = r_208.x;
                    acc_162[21] = r_208.y;
                    float2 _f2_140 = make_float2(acc_162[22], acc_162[23]);
                    float2 a_209 = _f2_140;
                    float2 _f2_141 = make_float2(_unpack_fp4_10[6], _unpack_fp4_10[7]);
                    float2 q_210 = _f2_141;
                    float2 r_211 = fma_f32x2_rn_noftz(q_210, scale1_175, a_209);
                    acc_162[22] = r_211.x;
                    acc_162[23] = r_211.y;
                    float _unpack_fp4_11[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_167[3 + 0]));
                        (&_unpack_fp4_11[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_11[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_11[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_11[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_11[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_11[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_11[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_11[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_11[0]), "=f"(_unpack_fp4_11[1]), "=f"(_unpack_fp4_11[2]), "=f"(_unpack_fp4_11[3]), "=f"(_unpack_fp4_11[4]), "=f"(_unpack_fp4_11[5]), "=f"(_unpack_fp4_11[6]), "=f"(_unpack_fp4_11[7]) : "r"(r_167[3 + 0]));
                        #endif
                    }
                    float2 _f2_142 = make_float2(acc_162[24], acc_162[25]);
                    float2 a_212 = _f2_142;
                    float2 _f2_143 = make_float2(_unpack_fp4_11[0], _unpack_fp4_11[1]);
                    float2 q_213 = _f2_143;
                    float2 r_214 = fma_f32x2_rn_noftz(q_213, scale1_175, a_212);
                    acc_162[24] = r_214.x;
                    acc_162[25] = r_214.y;
                    float2 _f2_144 = make_float2(acc_162[26], acc_162[27]);
                    float2 a_215 = _f2_144;
                    float2 _f2_145 = make_float2(_unpack_fp4_11[2], _unpack_fp4_11[3]);
                    float2 q_216 = _f2_145;
                    float2 r_217 = fma_f32x2_rn_noftz(q_216, scale1_175, a_215);
                    acc_162[26] = r_217.x;
                    acc_162[27] = r_217.y;
                    float2 _f2_146 = make_float2(acc_162[28], acc_162[29]);
                    float2 a_218 = _f2_146;
                    float2 _f2_147 = make_float2(_unpack_fp4_11[4], _unpack_fp4_11[5]);
                    float2 q_219 = _f2_147;
                    float2 r_220 = fma_f32x2_rn_noftz(q_219, scale1_175, a_218);
                    acc_162[28] = r_220.x;
                    acc_162[29] = r_220.y;
                    float2 _f2_148 = make_float2(acc_162[30], acc_162[31]);
                    float2 a_221 = _f2_148;
                    float2 _f2_149 = make_float2(_unpack_fp4_11[6], _unpack_fp4_11[7]);
                    float2 q_222 = _f2_149;
                    float2 r_223 = fma_f32x2_rn_noftz(q_222, scale1_175, a_221);
                    acc_162[30] = r_223.x;
                    acc_162[31] = r_223.y;
                    uint32_t _e4m3x2_to_f16x2_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_3) : "h"((uint16_t)(b_sc_161)));
                    float _cvt_f32_f16_6;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(_e4m3x2_to_f16x2_3)));
                    float _cvt_f32_f16_7;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(_e4m3x2_to_f16x2_3 >> 16)));
                    float _mul_6;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_cvt_f32_f16_6), "f"(rs_b));
                    float _mul_7;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_cvt_f32_f16_7), "f"(rs_b));
                    float lo_224 = _mul_6;
                    float hi_225 = _mul_7;
                    if (fhi_163) {
                        lo_224 = _mul_7;
                        hi_225 = _mul_6;
                    }
                    float2 _f2_150 = make_float2(lo_224, lo_224);
                    float2 scale0_226 = _f2_150;
                    float2 _f2_151 = make_float2(hi_225, hi_225);
                    float2 scale1_227 = _f2_151;
                    float _unpack_fp4_12[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_171[0]));
                        (&_unpack_fp4_12[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_12[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_12[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_12[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_12[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_12[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_12[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_12[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_12[0]), "=f"(_unpack_fp4_12[1]), "=f"(_unpack_fp4_12[2]), "=f"(_unpack_fp4_12[3]), "=f"(_unpack_fp4_12[4]), "=f"(_unpack_fp4_12[5]), "=f"(_unpack_fp4_12[6]), "=f"(_unpack_fp4_12[7]) : "r"(r_171[0]));
                        #endif
                    }
                    float2 _f2_152 = make_float2(acc_162[0], acc_162[1]);
                    float2 a_228 = _f2_152;
                    float2 _f2_153 = make_float2(_unpack_fp4_12[0], _unpack_fp4_12[1]);
                    float2 q_229 = _f2_153;
                    float2 r_230 = fma_f32x2_rn_noftz(q_229, scale0_226, a_228);
                    acc_162[0] = r_230.x;
                    acc_162[1] = r_230.y;
                    float2 _f2_154 = make_float2(acc_162[2], acc_162[3]);
                    float2 a_231 = _f2_154;
                    float2 _f2_155 = make_float2(_unpack_fp4_12[2], _unpack_fp4_12[3]);
                    float2 q_232 = _f2_155;
                    float2 r_233 = fma_f32x2_rn_noftz(q_232, scale0_226, a_231);
                    acc_162[2] = r_233.x;
                    acc_162[3] = r_233.y;
                    float2 _f2_156 = make_float2(acc_162[4], acc_162[5]);
                    float2 a_234 = _f2_156;
                    float2 _f2_157 = make_float2(_unpack_fp4_12[4], _unpack_fp4_12[5]);
                    float2 q_235 = _f2_157;
                    float2 r_236 = fma_f32x2_rn_noftz(q_235, scale0_226, a_234);
                    acc_162[4] = r_236.x;
                    acc_162[5] = r_236.y;
                    float2 _f2_158 = make_float2(acc_162[6], acc_162[7]);
                    float2 a_237 = _f2_158;
                    float2 _f2_159 = make_float2(_unpack_fp4_12[6], _unpack_fp4_12[7]);
                    float2 q_238 = _f2_159;
                    float2 r_239 = fma_f32x2_rn_noftz(q_238, scale0_226, a_237);
                    acc_162[6] = r_239.x;
                    acc_162[7] = r_239.y;
                    float _unpack_fp4_13[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_171[1 + 0]));
                        (&_unpack_fp4_13[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_13[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_13[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_13[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_13[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_13[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_13[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_13[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_13[0]), "=f"(_unpack_fp4_13[1]), "=f"(_unpack_fp4_13[2]), "=f"(_unpack_fp4_13[3]), "=f"(_unpack_fp4_13[4]), "=f"(_unpack_fp4_13[5]), "=f"(_unpack_fp4_13[6]), "=f"(_unpack_fp4_13[7]) : "r"(r_171[1 + 0]));
                        #endif
                    }
                    float2 _f2_160 = make_float2(acc_162[8], acc_162[9]);
                    float2 a_240 = _f2_160;
                    float2 _f2_161 = make_float2(_unpack_fp4_13[0], _unpack_fp4_13[1]);
                    float2 q_241 = _f2_161;
                    float2 r_242 = fma_f32x2_rn_noftz(q_241, scale0_226, a_240);
                    acc_162[8] = r_242.x;
                    acc_162[9] = r_242.y;
                    float2 _f2_162 = make_float2(acc_162[10], acc_162[11]);
                    float2 a_243 = _f2_162;
                    float2 _f2_163 = make_float2(_unpack_fp4_13[2], _unpack_fp4_13[3]);
                    float2 q_244 = _f2_163;
                    float2 r_245 = fma_f32x2_rn_noftz(q_244, scale0_226, a_243);
                    acc_162[10] = r_245.x;
                    acc_162[11] = r_245.y;
                    float2 _f2_164 = make_float2(acc_162[12], acc_162[13]);
                    float2 a_246 = _f2_164;
                    float2 _f2_165 = make_float2(_unpack_fp4_13[4], _unpack_fp4_13[5]);
                    float2 q_247 = _f2_165;
                    float2 r_248 = fma_f32x2_rn_noftz(q_247, scale0_226, a_246);
                    acc_162[12] = r_248.x;
                    acc_162[13] = r_248.y;
                    float2 _f2_166 = make_float2(acc_162[14], acc_162[15]);
                    float2 a_249 = _f2_166;
                    float2 _f2_167 = make_float2(_unpack_fp4_13[6], _unpack_fp4_13[7]);
                    float2 q_250 = _f2_167;
                    float2 r_251 = fma_f32x2_rn_noftz(q_250, scale0_226, a_249);
                    acc_162[14] = r_251.x;
                    acc_162[15] = r_251.y;
                    float _unpack_fp4_14[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_171[2 + 0]));
                        (&_unpack_fp4_14[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_14[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_14[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_14[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_14[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_14[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_14[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_14[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_14[0]), "=f"(_unpack_fp4_14[1]), "=f"(_unpack_fp4_14[2]), "=f"(_unpack_fp4_14[3]), "=f"(_unpack_fp4_14[4]), "=f"(_unpack_fp4_14[5]), "=f"(_unpack_fp4_14[6]), "=f"(_unpack_fp4_14[7]) : "r"(r_171[2 + 0]));
                        #endif
                    }
                    float2 _f2_168 = make_float2(acc_162[16], acc_162[17]);
                    float2 a_252 = _f2_168;
                    float2 _f2_169 = make_float2(_unpack_fp4_14[0], _unpack_fp4_14[1]);
                    float2 q_253 = _f2_169;
                    float2 r_254 = fma_f32x2_rn_noftz(q_253, scale1_227, a_252);
                    acc_162[16] = r_254.x;
                    acc_162[17] = r_254.y;
                    float2 _f2_170 = make_float2(acc_162[18], acc_162[19]);
                    float2 a_255 = _f2_170;
                    float2 _f2_171 = make_float2(_unpack_fp4_14[2], _unpack_fp4_14[3]);
                    float2 q_256 = _f2_171;
                    float2 r_257 = fma_f32x2_rn_noftz(q_256, scale1_227, a_255);
                    acc_162[18] = r_257.x;
                    acc_162[19] = r_257.y;
                    float2 _f2_172 = make_float2(acc_162[20], acc_162[21]);
                    float2 a_258 = _f2_172;
                    float2 _f2_173 = make_float2(_unpack_fp4_14[4], _unpack_fp4_14[5]);
                    float2 q_259 = _f2_173;
                    float2 r_260 = fma_f32x2_rn_noftz(q_259, scale1_227, a_258);
                    acc_162[20] = r_260.x;
                    acc_162[21] = r_260.y;
                    float2 _f2_174 = make_float2(acc_162[22], acc_162[23]);
                    float2 a_261 = _f2_174;
                    float2 _f2_175 = make_float2(_unpack_fp4_14[6], _unpack_fp4_14[7]);
                    float2 q_262 = _f2_175;
                    float2 r_263 = fma_f32x2_rn_noftz(q_262, scale1_227, a_261);
                    acc_162[22] = r_263.x;
                    acc_162[23] = r_263.y;
                    float _unpack_fp4_15[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_171[3 + 0]));
                        (&_unpack_fp4_15[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_15[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_15[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_15[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_15[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_15[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_15[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_15[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_15[0]), "=f"(_unpack_fp4_15[1]), "=f"(_unpack_fp4_15[2]), "=f"(_unpack_fp4_15[3]), "=f"(_unpack_fp4_15[4]), "=f"(_unpack_fp4_15[5]), "=f"(_unpack_fp4_15[6]), "=f"(_unpack_fp4_15[7]) : "r"(r_171[3 + 0]));
                        #endif
                    }
                    float2 _f2_176 = make_float2(acc_162[24], acc_162[25]);
                    float2 a_264 = _f2_176;
                    float2 _f2_177 = make_float2(_unpack_fp4_15[0], _unpack_fp4_15[1]);
                    float2 q_265 = _f2_177;
                    float2 r_266 = fma_f32x2_rn_noftz(q_265, scale1_227, a_264);
                    acc_162[24] = r_266.x;
                    acc_162[25] = r_266.y;
                    float2 _f2_178 = make_float2(acc_162[26], acc_162[27]);
                    float2 a_267 = _f2_178;
                    float2 _f2_179 = make_float2(_unpack_fp4_15[2], _unpack_fp4_15[3]);
                    float2 q_268 = _f2_179;
                    float2 r_269 = fma_f32x2_rn_noftz(q_268, scale1_227, a_267);
                    acc_162[26] = r_269.x;
                    acc_162[27] = r_269.y;
                    float2 _f2_180 = make_float2(acc_162[28], acc_162[29]);
                    float2 a_270 = _f2_180;
                    float2 _f2_181 = make_float2(_unpack_fp4_15[4], _unpack_fp4_15[5]);
                    float2 q_271 = _f2_181;
                    float2 r_272 = fma_f32x2_rn_noftz(q_271, scale1_227, a_270);
                    acc_162[28] = r_272.x;
                    acc_162[29] = r_272.y;
                    float2 _f2_182 = make_float2(acc_162[30], acc_162[31]);
                    float2 a_273 = _f2_182;
                    float2 _f2_183 = make_float2(_unpack_fp4_15[6], _unpack_fp4_15[7]);
                    float2 q_274 = _f2_183;
                    float2 r_275 = fma_f32x2_rn_noftz(q_274, scale1_227, a_273);
                    acc_162[30] = r_275.x;
                    acc_162[31] = r_275.y;
                    unsigned int t_276[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_276[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_276[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_276[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_276[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_276[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 3))));
                    float2 _f2_184 = make_float2(acc_162[0], acc_162[1]);
                    float2 a2_277 = _f2_184;
                    float2 _f2_185 = make_float2(__uint_as_float(t_276[0] << 16), __uint_as_float(t_276[0] & 4294901760u));
                    float2 z_278 = add_f32x2_noftz(a2_277, _f2_185);
                    acc_162[0] = z_278.x;
                    acc_162[1] = z_278.y;
                    float2 _f2_186 = make_float2(ssx, ssy);
                    float2 ss2_279 = fma_f32x2_rn_noftz(z_278, z_278, _f2_186);
                    ssx = ss2_279.x;
                    ssy = ss2_279.y;
                    float2 _f2_187 = make_float2(acc_162[2], acc_162[3]);
                    float2 a2_280 = _f2_187;
                    float2 _f2_188 = make_float2(__uint_as_float(t_276[1] << 16), __uint_as_float(t_276[1] & 4294901760u));
                    float2 z_281 = add_f32x2_noftz(a2_280, _f2_188);
                    acc_162[2] = z_281.x;
                    acc_162[3] = z_281.y;
                    float2 _f2_189 = make_float2(ssx, ssy);
                    float2 ss2_282 = fma_f32x2_rn_noftz(z_281, z_281, _f2_189);
                    ssx = ss2_282.x;
                    ssy = ss2_282.y;
                    float2 _f2_190 = make_float2(acc_162[4], acc_162[5]);
                    float2 a2_283 = _f2_190;
                    float2 _f2_191 = make_float2(__uint_as_float(t_276[2] << 16), __uint_as_float(t_276[2] & 4294901760u));
                    float2 z_284 = add_f32x2_noftz(a2_283, _f2_191);
                    acc_162[4] = z_284.x;
                    acc_162[5] = z_284.y;
                    float2 _f2_192 = make_float2(ssx, ssy);
                    float2 ss2_285 = fma_f32x2_rn_noftz(z_284, z_284, _f2_192);
                    ssx = ss2_285.x;
                    ssy = ss2_285.y;
                    float2 _f2_193 = make_float2(acc_162[6], acc_162[7]);
                    float2 a2_286 = _f2_193;
                    float2 _f2_194 = make_float2(__uint_as_float(t_276[3] << 16), __uint_as_float(t_276[3] & 4294901760u));
                    float2 z_287 = add_f32x2_noftz(a2_286, _f2_194);
                    acc_162[6] = z_287.x;
                    acc_162[7] = z_287.y;
                    float2 _f2_195 = make_float2(ssx, ssy);
                    float2 ss2_288 = fma_f32x2_rn_noftz(z_287, z_287, _f2_195);
                    ssx = ss2_288.x;
                    ssy = ss2_288.y;
                    float2 _f2_196 = make_float2(acc_162[8], acc_162[9]);
                    float2 a2_289 = _f2_196;
                    float2 _f2_197 = make_float2(__uint_as_float(t_276[4] << 16), __uint_as_float(t_276[4] & 4294901760u));
                    float2 z_290 = add_f32x2_noftz(a2_289, _f2_197);
                    acc_162[8] = z_290.x;
                    acc_162[9] = z_290.y;
                    float2 _f2_198 = make_float2(ssx, ssy);
                    float2 ss2_291 = fma_f32x2_rn_noftz(z_290, z_290, _f2_198);
                    ssx = ss2_291.x;
                    ssy = ss2_291.y;
                    float2 _f2_199 = make_float2(acc_162[10], acc_162[11]);
                    float2 a2_292 = _f2_199;
                    float2 _f2_200 = make_float2(__uint_as_float(t_276[5] << 16), __uint_as_float(t_276[5] & 4294901760u));
                    float2 z_293 = add_f32x2_noftz(a2_292, _f2_200);
                    acc_162[10] = z_293.x;
                    acc_162[11] = z_293.y;
                    float2 _f2_201 = make_float2(ssx, ssy);
                    float2 ss2_294 = fma_f32x2_rn_noftz(z_293, z_293, _f2_201);
                    ssx = ss2_294.x;
                    ssy = ss2_294.y;
                    float2 _f2_202 = make_float2(acc_162[12], acc_162[13]);
                    float2 a2_295 = _f2_202;
                    float2 _f2_203 = make_float2(__uint_as_float(t_276[6] << 16), __uint_as_float(t_276[6] & 4294901760u));
                    float2 z_296 = add_f32x2_noftz(a2_295, _f2_203);
                    acc_162[12] = z_296.x;
                    acc_162[13] = z_296.y;
                    float2 _f2_204 = make_float2(ssx, ssy);
                    float2 ss2_297 = fma_f32x2_rn_noftz(z_296, z_296, _f2_204);
                    ssx = ss2_297.x;
                    ssy = ss2_297.y;
                    float2 _f2_205 = make_float2(acc_162[14], acc_162[15]);
                    float2 a2_298 = _f2_205;
                    float2 _f2_206 = make_float2(__uint_as_float(t_276[7] << 16), __uint_as_float(t_276[7] & 4294901760u));
                    float2 z_299 = add_f32x2_noftz(a2_298, _f2_206);
                    acc_162[14] = z_299.x;
                    acc_162[15] = z_299.y;
                    float2 _f2_207 = make_float2(ssx, ssy);
                    float2 ss2_300 = fma_f32x2_rn_noftz(z_299, z_299, _f2_207);
                    ssx = ss2_300.x;
                    ssy = ss2_300.y;
                    float2 _f2_208 = make_float2(acc_162[16], acc_162[17]);
                    float2 a2_301 = _f2_208;
                    float2 _f2_209 = make_float2(__uint_as_float(t_276[8] << 16), __uint_as_float(t_276[8] & 4294901760u));
                    float2 z_302 = add_f32x2_noftz(a2_301, _f2_209);
                    acc_162[16] = z_302.x;
                    acc_162[17] = z_302.y;
                    float2 _f2_210 = make_float2(ssx, ssy);
                    float2 ss2_303 = fma_f32x2_rn_noftz(z_302, z_302, _f2_210);
                    ssx = ss2_303.x;
                    ssy = ss2_303.y;
                    float2 _f2_211 = make_float2(acc_162[18], acc_162[19]);
                    float2 a2_304 = _f2_211;
                    float2 _f2_212 = make_float2(__uint_as_float(t_276[9] << 16), __uint_as_float(t_276[9] & 4294901760u));
                    float2 z_305 = add_f32x2_noftz(a2_304, _f2_212);
                    acc_162[18] = z_305.x;
                    acc_162[19] = z_305.y;
                    float2 _f2_213 = make_float2(ssx, ssy);
                    float2 ss2_306 = fma_f32x2_rn_noftz(z_305, z_305, _f2_213);
                    ssx = ss2_306.x;
                    ssy = ss2_306.y;
                    float2 _f2_214 = make_float2(acc_162[20], acc_162[21]);
                    float2 a2_307 = _f2_214;
                    float2 _f2_215 = make_float2(__uint_as_float(t_276[10] << 16), __uint_as_float(t_276[10] & 4294901760u));
                    float2 z_308 = add_f32x2_noftz(a2_307, _f2_215);
                    acc_162[20] = z_308.x;
                    acc_162[21] = z_308.y;
                    float2 _f2_216 = make_float2(ssx, ssy);
                    float2 ss2_309 = fma_f32x2_rn_noftz(z_308, z_308, _f2_216);
                    ssx = ss2_309.x;
                    ssy = ss2_309.y;
                    float2 _f2_217 = make_float2(acc_162[22], acc_162[23]);
                    float2 a2_310 = _f2_217;
                    float2 _f2_218 = make_float2(__uint_as_float(t_276[11] << 16), __uint_as_float(t_276[11] & 4294901760u));
                    float2 z_311 = add_f32x2_noftz(a2_310, _f2_218);
                    acc_162[22] = z_311.x;
                    acc_162[23] = z_311.y;
                    float2 _f2_219 = make_float2(ssx, ssy);
                    float2 ss2_312 = fma_f32x2_rn_noftz(z_311, z_311, _f2_219);
                    ssx = ss2_312.x;
                    ssy = ss2_312.y;
                    float2 _f2_220 = make_float2(acc_162[24], acc_162[25]);
                    float2 a2_313 = _f2_220;
                    float2 _f2_221 = make_float2(__uint_as_float(t_276[12] << 16), __uint_as_float(t_276[12] & 4294901760u));
                    float2 z_314 = add_f32x2_noftz(a2_313, _f2_221);
                    acc_162[24] = z_314.x;
                    acc_162[25] = z_314.y;
                    float2 _f2_222 = make_float2(ssx, ssy);
                    float2 ss2_315 = fma_f32x2_rn_noftz(z_314, z_314, _f2_222);
                    ssx = ss2_315.x;
                    ssy = ss2_315.y;
                    float2 _f2_223 = make_float2(acc_162[26], acc_162[27]);
                    float2 a2_316 = _f2_223;
                    float2 _f2_224 = make_float2(__uint_as_float(t_276[13] << 16), __uint_as_float(t_276[13] & 4294901760u));
                    float2 z_317 = add_f32x2_noftz(a2_316, _f2_224);
                    acc_162[26] = z_317.x;
                    acc_162[27] = z_317.y;
                    float2 _f2_225 = make_float2(ssx, ssy);
                    float2 ss2_318 = fma_f32x2_rn_noftz(z_317, z_317, _f2_225);
                    ssx = ss2_318.x;
                    ssy = ss2_318.y;
                    float2 _f2_226 = make_float2(acc_162[28], acc_162[29]);
                    float2 a2_319 = _f2_226;
                    float2 _f2_227 = make_float2(__uint_as_float(t_276[14] << 16), __uint_as_float(t_276[14] & 4294901760u));
                    float2 z_320 = add_f32x2_noftz(a2_319, _f2_227);
                    acc_162[28] = z_320.x;
                    acc_162[29] = z_320.y;
                    float2 _f2_228 = make_float2(ssx, ssy);
                    float2 ss2_321 = fma_f32x2_rn_noftz(z_320, z_320, _f2_228);
                    ssx = ss2_321.x;
                    ssy = ss2_321.y;
                    float2 _f2_229 = make_float2(acc_162[30], acc_162[31]);
                    float2 a2_322 = _f2_229;
                    float2 _f2_230 = make_float2(__uint_as_float(t_276[15] << 16), __uint_as_float(t_276[15] & 4294901760u));
                    float2 z_323 = add_f32x2_noftz(a2_322, _f2_230);
                    acc_162[30] = z_323.x;
                    acc_162[31] = z_323.y;
                    float2 _f2_231 = make_float2(ssx, ssy);
                    float2 ss2_324 = fma_f32x2_rn_noftz(z_323, z_323, _f2_231);
                    ssx = ss2_324.x;
                    ssy = ss2_324.y;
                    unsigned int acc_162_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_162[_lp*2 + 0], acc_162[_lp*2+1 + 0]));
                        acc_162_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 0))), "r"(acc_162_bf16[0]), "r"(acc_162_bf16[1]), "r"(acc_162_bf16[2]), "r"(acc_162_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 1))), "r"(acc_162_bf16[4]), "r"(acc_162_bf16[5]), "r"(acc_162_bf16[6]), "r"(acc_162_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 2))), "r"(acc_162_bf16[8]), "r"(acc_162_bf16[9]), "r"(acc_162_bf16[10]), "r"(acc_162_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_155 + 16 * (f ^ 3))), "r"(acc_162_bf16[12]), "r"(acc_162_bf16[13]), "r"(acc_162_bf16[14]), "r"(acc_162_bf16[15]) : "memory");
                    unsigned int d_325 = 0;
                    d_325 = d_325 | pv[8] & ~((pv[8] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[9] & ~((pv[9] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[10] & ~((pv[10] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[11] & ~((pv[11] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | ((ps[2] & 255) + 1 | (ps[2] >> 8) + 1) & 256;
                    d_325 = d_325 | pv[12] & ~((pv[12] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[13] & ~((pv[13] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[14] & ~((pv[14] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | pv[15] & ~((pv[15] & 2004318071) + 2004318071) & 2290649224u;
                    d_325 = d_325 | ((ps[3] & 255) + 1 | (ps[3] >> 8) + 1) & 256;
                    if (d_325 != 0) {
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
                    int pp_326 = lane_2 + 64;
                    unsigned int own_327[4];
                    unsigned int own_sc_328[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_327[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_327[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_327[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_327[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_326)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_328[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_326)));
                    unsigned int a_329[4];
                    unsigned int b_330[4];
                    a_329[0] = ((first) ? own_327[0] : pv[8]);
                    b_330[0] = ((first) ? pv[8] : own_327[0]);
                    a_329[1] = ((first) ? own_327[1] : pv[9]);
                    b_330[1] = ((first) ? pv[9] : own_327[1]);
                    a_329[2] = ((first) ? own_327[2] : pv[10]);
                    b_330[2] = ((first) ? pv[10] : own_327[2]);
                    a_329[3] = ((first) ? own_327[3] : pv[11]);
                    b_330[3] = ((first) ? pv[11] : own_327[3]);
                    unsigned int a_sc_331 = ((first) ? own_sc_328[0] : ps[2]);
                    unsigned int b_sc_332 = ((first) ? ps[2] : own_sc_328[0]);
                    float acc_333[32] = {0};
                    bool fhi_334 = (f & 2) != 0;
                    bool fodd_335 = (f & 1) != 0;
                    bool fhi_336 = (f & 2) != 0;
                    unsigned int s_337[4];
                    s_337[0] = a_329[0];
                    s_337[1] = a_329[1];
                    s_337[2] = a_329[2];
                    s_337[3] = a_329[3];
                    if (fodd_335) {
                        s_337[0] = a_329[1];
                        s_337[1] = a_329[0];
                        s_337[2] = a_329[3];
                        s_337[3] = a_329[2];
                    }
                    unsigned int r_338[4];
                    r_338[0] = s_337[0];
                    r_338[1] = s_337[1];
                    r_338[2] = s_337[2];
                    r_338[3] = s_337[3];
                    if (fhi_336) {
                        r_338[0] = s_337[2];
                        r_338[1] = s_337[3];
                        r_338[2] = s_337[0];
                        r_338[3] = s_337[1];
                    }
                    bool fodd_339 = (f & 1) != 0;
                    bool fhi_340 = (f & 2) != 0;
                    unsigned int s_341[4];
                    s_341[0] = b_330[0];
                    s_341[1] = b_330[1];
                    s_341[2] = b_330[2];
                    s_341[3] = b_330[3];
                    if (fodd_339) {
                        s_341[0] = b_330[1];
                        s_341[1] = b_330[0];
                        s_341[2] = b_330[3];
                        s_341[3] = b_330[2];
                    }
                    unsigned int r_342[4];
                    r_342[0] = s_341[0];
                    r_342[1] = s_341[1];
                    r_342[2] = s_341[2];
                    r_342[3] = s_341[3];
                    if (fhi_340) {
                        r_342[0] = s_341[2];
                        r_342[1] = s_341[3];
                        r_342[2] = s_341[0];
                        r_342[3] = s_341[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_4) : "h"((uint16_t)(a_sc_331)));
                    float _cvt_f32_f16_8;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(_e4m3x2_to_f16x2_4)));
                    float _cvt_f32_f16_9;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(_e4m3x2_to_f16x2_4 >> 16)));
                    float _mul_8;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_cvt_f32_f16_8), "f"(rs_a));
                    float _mul_9;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_cvt_f32_f16_9), "f"(rs_a));
                    float lo_343 = _mul_8;
                    float hi_344 = _mul_9;
                    if (fhi_334) {
                        lo_343 = _mul_9;
                        hi_344 = _mul_8;
                    }
                    float2 _f2_232 = make_float2(lo_343, lo_343);
                    float2 scale0_345 = _f2_232;
                    float2 _f2_233 = make_float2(hi_344, hi_344);
                    float2 scale1_346 = _f2_233;
                    float _unpack_fp4_16[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_338[0]));
                        (&_unpack_fp4_16[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_16[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_16[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_16[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_16[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_16[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_16[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_16[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_16[0]), "=f"(_unpack_fp4_16[1]), "=f"(_unpack_fp4_16[2]), "=f"(_unpack_fp4_16[3]), "=f"(_unpack_fp4_16[4]), "=f"(_unpack_fp4_16[5]), "=f"(_unpack_fp4_16[6]), "=f"(_unpack_fp4_16[7]) : "r"(r_338[0]));
                        #endif
                    }
                    float2 _f2_234 = make_float2(acc_333[0], acc_333[1]);
                    float2 a_347 = _f2_234;
                    float2 _f2_235 = make_float2(_unpack_fp4_16[0], _unpack_fp4_16[1]);
                    float2 q_348 = _f2_235;
                    float2 r_349 = fma_f32x2_rn_noftz(q_348, scale0_345, a_347);
                    acc_333[0] = r_349.x;
                    acc_333[1] = r_349.y;
                    float2 _f2_236 = make_float2(acc_333[2], acc_333[3]);
                    float2 a_350 = _f2_236;
                    float2 _f2_237 = make_float2(_unpack_fp4_16[2], _unpack_fp4_16[3]);
                    float2 q_351 = _f2_237;
                    float2 r_352 = fma_f32x2_rn_noftz(q_351, scale0_345, a_350);
                    acc_333[2] = r_352.x;
                    acc_333[3] = r_352.y;
                    float2 _f2_238 = make_float2(acc_333[4], acc_333[5]);
                    float2 a_353 = _f2_238;
                    float2 _f2_239 = make_float2(_unpack_fp4_16[4], _unpack_fp4_16[5]);
                    float2 q_354 = _f2_239;
                    float2 r_355 = fma_f32x2_rn_noftz(q_354, scale0_345, a_353);
                    acc_333[4] = r_355.x;
                    acc_333[5] = r_355.y;
                    float2 _f2_240 = make_float2(acc_333[6], acc_333[7]);
                    float2 a_356 = _f2_240;
                    float2 _f2_241 = make_float2(_unpack_fp4_16[6], _unpack_fp4_16[7]);
                    float2 q_357 = _f2_241;
                    float2 r_358 = fma_f32x2_rn_noftz(q_357, scale0_345, a_356);
                    acc_333[6] = r_358.x;
                    acc_333[7] = r_358.y;
                    float _unpack_fp4_17[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_338[1 + 0]));
                        (&_unpack_fp4_17[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_17[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_17[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_17[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_17[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_17[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_17[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_17[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_17[0]), "=f"(_unpack_fp4_17[1]), "=f"(_unpack_fp4_17[2]), "=f"(_unpack_fp4_17[3]), "=f"(_unpack_fp4_17[4]), "=f"(_unpack_fp4_17[5]), "=f"(_unpack_fp4_17[6]), "=f"(_unpack_fp4_17[7]) : "r"(r_338[1 + 0]));
                        #endif
                    }
                    float2 _f2_242 = make_float2(acc_333[8], acc_333[9]);
                    float2 a_359 = _f2_242;
                    float2 _f2_243 = make_float2(_unpack_fp4_17[0], _unpack_fp4_17[1]);
                    float2 q_360 = _f2_243;
                    float2 r_361 = fma_f32x2_rn_noftz(q_360, scale0_345, a_359);
                    acc_333[8] = r_361.x;
                    acc_333[9] = r_361.y;
                    float2 _f2_244 = make_float2(acc_333[10], acc_333[11]);
                    float2 a_362 = _f2_244;
                    float2 _f2_245 = make_float2(_unpack_fp4_17[2], _unpack_fp4_17[3]);
                    float2 q_363 = _f2_245;
                    float2 r_364 = fma_f32x2_rn_noftz(q_363, scale0_345, a_362);
                    acc_333[10] = r_364.x;
                    acc_333[11] = r_364.y;
                    float2 _f2_246 = make_float2(acc_333[12], acc_333[13]);
                    float2 a_365 = _f2_246;
                    float2 _f2_247 = make_float2(_unpack_fp4_17[4], _unpack_fp4_17[5]);
                    float2 q_366 = _f2_247;
                    float2 r_367 = fma_f32x2_rn_noftz(q_366, scale0_345, a_365);
                    acc_333[12] = r_367.x;
                    acc_333[13] = r_367.y;
                    float2 _f2_248 = make_float2(acc_333[14], acc_333[15]);
                    float2 a_368 = _f2_248;
                    float2 _f2_249 = make_float2(_unpack_fp4_17[6], _unpack_fp4_17[7]);
                    float2 q_369 = _f2_249;
                    float2 r_370 = fma_f32x2_rn_noftz(q_369, scale0_345, a_368);
                    acc_333[14] = r_370.x;
                    acc_333[15] = r_370.y;
                    float _unpack_fp4_18[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_338[2 + 0]));
                        (&_unpack_fp4_18[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_18[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_18[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_18[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_18[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_18[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_18[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_18[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_18[0]), "=f"(_unpack_fp4_18[1]), "=f"(_unpack_fp4_18[2]), "=f"(_unpack_fp4_18[3]), "=f"(_unpack_fp4_18[4]), "=f"(_unpack_fp4_18[5]), "=f"(_unpack_fp4_18[6]), "=f"(_unpack_fp4_18[7]) : "r"(r_338[2 + 0]));
                        #endif
                    }
                    float2 _f2_250 = make_float2(acc_333[16], acc_333[17]);
                    float2 a_371 = _f2_250;
                    float2 _f2_251 = make_float2(_unpack_fp4_18[0], _unpack_fp4_18[1]);
                    float2 q_372 = _f2_251;
                    float2 r_373 = fma_f32x2_rn_noftz(q_372, scale1_346, a_371);
                    acc_333[16] = r_373.x;
                    acc_333[17] = r_373.y;
                    float2 _f2_252 = make_float2(acc_333[18], acc_333[19]);
                    float2 a_374 = _f2_252;
                    float2 _f2_253 = make_float2(_unpack_fp4_18[2], _unpack_fp4_18[3]);
                    float2 q_375 = _f2_253;
                    float2 r_376 = fma_f32x2_rn_noftz(q_375, scale1_346, a_374);
                    acc_333[18] = r_376.x;
                    acc_333[19] = r_376.y;
                    float2 _f2_254 = make_float2(acc_333[20], acc_333[21]);
                    float2 a_377 = _f2_254;
                    float2 _f2_255 = make_float2(_unpack_fp4_18[4], _unpack_fp4_18[5]);
                    float2 q_378 = _f2_255;
                    float2 r_379 = fma_f32x2_rn_noftz(q_378, scale1_346, a_377);
                    acc_333[20] = r_379.x;
                    acc_333[21] = r_379.y;
                    float2 _f2_256 = make_float2(acc_333[22], acc_333[23]);
                    float2 a_380 = _f2_256;
                    float2 _f2_257 = make_float2(_unpack_fp4_18[6], _unpack_fp4_18[7]);
                    float2 q_381 = _f2_257;
                    float2 r_382 = fma_f32x2_rn_noftz(q_381, scale1_346, a_380);
                    acc_333[22] = r_382.x;
                    acc_333[23] = r_382.y;
                    float _unpack_fp4_19[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_338[3 + 0]));
                        (&_unpack_fp4_19[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_19[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_19[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_19[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_19[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_19[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_19[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_19[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_19[0]), "=f"(_unpack_fp4_19[1]), "=f"(_unpack_fp4_19[2]), "=f"(_unpack_fp4_19[3]), "=f"(_unpack_fp4_19[4]), "=f"(_unpack_fp4_19[5]), "=f"(_unpack_fp4_19[6]), "=f"(_unpack_fp4_19[7]) : "r"(r_338[3 + 0]));
                        #endif
                    }
                    float2 _f2_258 = make_float2(acc_333[24], acc_333[25]);
                    float2 a_383 = _f2_258;
                    float2 _f2_259 = make_float2(_unpack_fp4_19[0], _unpack_fp4_19[1]);
                    float2 q_384 = _f2_259;
                    float2 r_385 = fma_f32x2_rn_noftz(q_384, scale1_346, a_383);
                    acc_333[24] = r_385.x;
                    acc_333[25] = r_385.y;
                    float2 _f2_260 = make_float2(acc_333[26], acc_333[27]);
                    float2 a_386 = _f2_260;
                    float2 _f2_261 = make_float2(_unpack_fp4_19[2], _unpack_fp4_19[3]);
                    float2 q_387 = _f2_261;
                    float2 r_388 = fma_f32x2_rn_noftz(q_387, scale1_346, a_386);
                    acc_333[26] = r_388.x;
                    acc_333[27] = r_388.y;
                    float2 _f2_262 = make_float2(acc_333[28], acc_333[29]);
                    float2 a_389 = _f2_262;
                    float2 _f2_263 = make_float2(_unpack_fp4_19[4], _unpack_fp4_19[5]);
                    float2 q_390 = _f2_263;
                    float2 r_391 = fma_f32x2_rn_noftz(q_390, scale1_346, a_389);
                    acc_333[28] = r_391.x;
                    acc_333[29] = r_391.y;
                    float2 _f2_264 = make_float2(acc_333[30], acc_333[31]);
                    float2 a_392 = _f2_264;
                    float2 _f2_265 = make_float2(_unpack_fp4_19[6], _unpack_fp4_19[7]);
                    float2 q_393 = _f2_265;
                    float2 r_394 = fma_f32x2_rn_noftz(q_393, scale1_346, a_392);
                    acc_333[30] = r_394.x;
                    acc_333[31] = r_394.y;
                    uint32_t _e4m3x2_to_f16x2_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_5) : "h"((uint16_t)(b_sc_332)));
                    float _cvt_f32_f16_10;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(_e4m3x2_to_f16x2_5)));
                    float _cvt_f32_f16_11;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(_e4m3x2_to_f16x2_5 >> 16)));
                    float _mul_10;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_cvt_f32_f16_10), "f"(rs_b));
                    float _mul_11;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_cvt_f32_f16_11), "f"(rs_b));
                    float lo_395 = _mul_10;
                    float hi_396 = _mul_11;
                    if (fhi_334) {
                        lo_395 = _mul_11;
                        hi_396 = _mul_10;
                    }
                    float2 _f2_266 = make_float2(lo_395, lo_395);
                    float2 scale0_397 = _f2_266;
                    float2 _f2_267 = make_float2(hi_396, hi_396);
                    float2 scale1_398 = _f2_267;
                    float _unpack_fp4_20[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_342[0]));
                        (&_unpack_fp4_20[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_20[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_20[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_20[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_20[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_20[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_20[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_20[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_20[0]), "=f"(_unpack_fp4_20[1]), "=f"(_unpack_fp4_20[2]), "=f"(_unpack_fp4_20[3]), "=f"(_unpack_fp4_20[4]), "=f"(_unpack_fp4_20[5]), "=f"(_unpack_fp4_20[6]), "=f"(_unpack_fp4_20[7]) : "r"(r_342[0]));
                        #endif
                    }
                    float2 _f2_268 = make_float2(acc_333[0], acc_333[1]);
                    float2 a_399 = _f2_268;
                    float2 _f2_269 = make_float2(_unpack_fp4_20[0], _unpack_fp4_20[1]);
                    float2 q_400 = _f2_269;
                    float2 r_401 = fma_f32x2_rn_noftz(q_400, scale0_397, a_399);
                    acc_333[0] = r_401.x;
                    acc_333[1] = r_401.y;
                    float2 _f2_270 = make_float2(acc_333[2], acc_333[3]);
                    float2 a_402 = _f2_270;
                    float2 _f2_271 = make_float2(_unpack_fp4_20[2], _unpack_fp4_20[3]);
                    float2 q_403 = _f2_271;
                    float2 r_404 = fma_f32x2_rn_noftz(q_403, scale0_397, a_402);
                    acc_333[2] = r_404.x;
                    acc_333[3] = r_404.y;
                    float2 _f2_272 = make_float2(acc_333[4], acc_333[5]);
                    float2 a_405 = _f2_272;
                    float2 _f2_273 = make_float2(_unpack_fp4_20[4], _unpack_fp4_20[5]);
                    float2 q_406 = _f2_273;
                    float2 r_407 = fma_f32x2_rn_noftz(q_406, scale0_397, a_405);
                    acc_333[4] = r_407.x;
                    acc_333[5] = r_407.y;
                    float2 _f2_274 = make_float2(acc_333[6], acc_333[7]);
                    float2 a_408 = _f2_274;
                    float2 _f2_275 = make_float2(_unpack_fp4_20[6], _unpack_fp4_20[7]);
                    float2 q_409 = _f2_275;
                    float2 r_410 = fma_f32x2_rn_noftz(q_409, scale0_397, a_408);
                    acc_333[6] = r_410.x;
                    acc_333[7] = r_410.y;
                    float _unpack_fp4_21[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_342[1 + 0]));
                        (&_unpack_fp4_21[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_21[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_21[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_21[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_21[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_21[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_21[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_21[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_21[0]), "=f"(_unpack_fp4_21[1]), "=f"(_unpack_fp4_21[2]), "=f"(_unpack_fp4_21[3]), "=f"(_unpack_fp4_21[4]), "=f"(_unpack_fp4_21[5]), "=f"(_unpack_fp4_21[6]), "=f"(_unpack_fp4_21[7]) : "r"(r_342[1 + 0]));
                        #endif
                    }
                    float2 _f2_276 = make_float2(acc_333[8], acc_333[9]);
                    float2 a_411 = _f2_276;
                    float2 _f2_277 = make_float2(_unpack_fp4_21[0], _unpack_fp4_21[1]);
                    float2 q_412 = _f2_277;
                    float2 r_413 = fma_f32x2_rn_noftz(q_412, scale0_397, a_411);
                    acc_333[8] = r_413.x;
                    acc_333[9] = r_413.y;
                    float2 _f2_278 = make_float2(acc_333[10], acc_333[11]);
                    float2 a_414 = _f2_278;
                    float2 _f2_279 = make_float2(_unpack_fp4_21[2], _unpack_fp4_21[3]);
                    float2 q_415 = _f2_279;
                    float2 r_416 = fma_f32x2_rn_noftz(q_415, scale0_397, a_414);
                    acc_333[10] = r_416.x;
                    acc_333[11] = r_416.y;
                    float2 _f2_280 = make_float2(acc_333[12], acc_333[13]);
                    float2 a_417 = _f2_280;
                    float2 _f2_281 = make_float2(_unpack_fp4_21[4], _unpack_fp4_21[5]);
                    float2 q_418 = _f2_281;
                    float2 r_419 = fma_f32x2_rn_noftz(q_418, scale0_397, a_417);
                    acc_333[12] = r_419.x;
                    acc_333[13] = r_419.y;
                    float2 _f2_282 = make_float2(acc_333[14], acc_333[15]);
                    float2 a_420 = _f2_282;
                    float2 _f2_283 = make_float2(_unpack_fp4_21[6], _unpack_fp4_21[7]);
                    float2 q_421 = _f2_283;
                    float2 r_422 = fma_f32x2_rn_noftz(q_421, scale0_397, a_420);
                    acc_333[14] = r_422.x;
                    acc_333[15] = r_422.y;
                    float _unpack_fp4_22[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_342[2 + 0]));
                        (&_unpack_fp4_22[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_22[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_22[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_22[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_22[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_22[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_22[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_22[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_22[0]), "=f"(_unpack_fp4_22[1]), "=f"(_unpack_fp4_22[2]), "=f"(_unpack_fp4_22[3]), "=f"(_unpack_fp4_22[4]), "=f"(_unpack_fp4_22[5]), "=f"(_unpack_fp4_22[6]), "=f"(_unpack_fp4_22[7]) : "r"(r_342[2 + 0]));
                        #endif
                    }
                    float2 _f2_284 = make_float2(acc_333[16], acc_333[17]);
                    float2 a_423 = _f2_284;
                    float2 _f2_285 = make_float2(_unpack_fp4_22[0], _unpack_fp4_22[1]);
                    float2 q_424 = _f2_285;
                    float2 r_425 = fma_f32x2_rn_noftz(q_424, scale1_398, a_423);
                    acc_333[16] = r_425.x;
                    acc_333[17] = r_425.y;
                    float2 _f2_286 = make_float2(acc_333[18], acc_333[19]);
                    float2 a_426 = _f2_286;
                    float2 _f2_287 = make_float2(_unpack_fp4_22[2], _unpack_fp4_22[3]);
                    float2 q_427 = _f2_287;
                    float2 r_428 = fma_f32x2_rn_noftz(q_427, scale1_398, a_426);
                    acc_333[18] = r_428.x;
                    acc_333[19] = r_428.y;
                    float2 _f2_288 = make_float2(acc_333[20], acc_333[21]);
                    float2 a_429 = _f2_288;
                    float2 _f2_289 = make_float2(_unpack_fp4_22[4], _unpack_fp4_22[5]);
                    float2 q_430 = _f2_289;
                    float2 r_431 = fma_f32x2_rn_noftz(q_430, scale1_398, a_429);
                    acc_333[20] = r_431.x;
                    acc_333[21] = r_431.y;
                    float2 _f2_290 = make_float2(acc_333[22], acc_333[23]);
                    float2 a_432 = _f2_290;
                    float2 _f2_291 = make_float2(_unpack_fp4_22[6], _unpack_fp4_22[7]);
                    float2 q_433 = _f2_291;
                    float2 r_434 = fma_f32x2_rn_noftz(q_433, scale1_398, a_432);
                    acc_333[22] = r_434.x;
                    acc_333[23] = r_434.y;
                    float _unpack_fp4_23[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_342[3 + 0]));
                        (&_unpack_fp4_23[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_23[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_23[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_23[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_23[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_23[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_23[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_23[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_23[0]), "=f"(_unpack_fp4_23[1]), "=f"(_unpack_fp4_23[2]), "=f"(_unpack_fp4_23[3]), "=f"(_unpack_fp4_23[4]), "=f"(_unpack_fp4_23[5]), "=f"(_unpack_fp4_23[6]), "=f"(_unpack_fp4_23[7]) : "r"(r_342[3 + 0]));
                        #endif
                    }
                    float2 _f2_292 = make_float2(acc_333[24], acc_333[25]);
                    float2 a_435 = _f2_292;
                    float2 _f2_293 = make_float2(_unpack_fp4_23[0], _unpack_fp4_23[1]);
                    float2 q_436 = _f2_293;
                    float2 r_437 = fma_f32x2_rn_noftz(q_436, scale1_398, a_435);
                    acc_333[24] = r_437.x;
                    acc_333[25] = r_437.y;
                    float2 _f2_294 = make_float2(acc_333[26], acc_333[27]);
                    float2 a_438 = _f2_294;
                    float2 _f2_295 = make_float2(_unpack_fp4_23[2], _unpack_fp4_23[3]);
                    float2 q_439 = _f2_295;
                    float2 r_440 = fma_f32x2_rn_noftz(q_439, scale1_398, a_438);
                    acc_333[26] = r_440.x;
                    acc_333[27] = r_440.y;
                    float2 _f2_296 = make_float2(acc_333[28], acc_333[29]);
                    float2 a_441 = _f2_296;
                    float2 _f2_297 = make_float2(_unpack_fp4_23[4], _unpack_fp4_23[5]);
                    float2 q_442 = _f2_297;
                    float2 r_443 = fma_f32x2_rn_noftz(q_442, scale1_398, a_441);
                    acc_333[28] = r_443.x;
                    acc_333[29] = r_443.y;
                    float2 _f2_298 = make_float2(acc_333[30], acc_333[31]);
                    float2 a_444 = _f2_298;
                    float2 _f2_299 = make_float2(_unpack_fp4_23[6], _unpack_fp4_23[7]);
                    float2 q_445 = _f2_299;
                    float2 r_446 = fma_f32x2_rn_noftz(q_445, scale1_398, a_444);
                    acc_333[30] = r_446.x;
                    acc_333[31] = r_446.y;
                    unsigned int t_447[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_447[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_447[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_447[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_447[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_447[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 3))));
                    float2 _f2_300 = make_float2(acc_333[0], acc_333[1]);
                    float2 a2_448 = _f2_300;
                    float2 _f2_301 = make_float2(__uint_as_float(t_447[0] << 16), __uint_as_float(t_447[0] & 4294901760u));
                    float2 z_449 = add_f32x2_noftz(a2_448, _f2_301);
                    acc_333[0] = z_449.x;
                    acc_333[1] = z_449.y;
                    float2 _f2_302 = make_float2(ssx, ssy);
                    float2 ss2_450 = fma_f32x2_rn_noftz(z_449, z_449, _f2_302);
                    ssx = ss2_450.x;
                    ssy = ss2_450.y;
                    float2 _f2_303 = make_float2(acc_333[2], acc_333[3]);
                    float2 a2_451 = _f2_303;
                    float2 _f2_304 = make_float2(__uint_as_float(t_447[1] << 16), __uint_as_float(t_447[1] & 4294901760u));
                    float2 z_452 = add_f32x2_noftz(a2_451, _f2_304);
                    acc_333[2] = z_452.x;
                    acc_333[3] = z_452.y;
                    float2 _f2_305 = make_float2(ssx, ssy);
                    float2 ss2_453 = fma_f32x2_rn_noftz(z_452, z_452, _f2_305);
                    ssx = ss2_453.x;
                    ssy = ss2_453.y;
                    float2 _f2_306 = make_float2(acc_333[4], acc_333[5]);
                    float2 a2_454 = _f2_306;
                    float2 _f2_307 = make_float2(__uint_as_float(t_447[2] << 16), __uint_as_float(t_447[2] & 4294901760u));
                    float2 z_455 = add_f32x2_noftz(a2_454, _f2_307);
                    acc_333[4] = z_455.x;
                    acc_333[5] = z_455.y;
                    float2 _f2_308 = make_float2(ssx, ssy);
                    float2 ss2_456 = fma_f32x2_rn_noftz(z_455, z_455, _f2_308);
                    ssx = ss2_456.x;
                    ssy = ss2_456.y;
                    float2 _f2_309 = make_float2(acc_333[6], acc_333[7]);
                    float2 a2_457 = _f2_309;
                    float2 _f2_310 = make_float2(__uint_as_float(t_447[3] << 16), __uint_as_float(t_447[3] & 4294901760u));
                    float2 z_458 = add_f32x2_noftz(a2_457, _f2_310);
                    acc_333[6] = z_458.x;
                    acc_333[7] = z_458.y;
                    float2 _f2_311 = make_float2(ssx, ssy);
                    float2 ss2_459 = fma_f32x2_rn_noftz(z_458, z_458, _f2_311);
                    ssx = ss2_459.x;
                    ssy = ss2_459.y;
                    float2 _f2_312 = make_float2(acc_333[8], acc_333[9]);
                    float2 a2_460 = _f2_312;
                    float2 _f2_313 = make_float2(__uint_as_float(t_447[4] << 16), __uint_as_float(t_447[4] & 4294901760u));
                    float2 z_461 = add_f32x2_noftz(a2_460, _f2_313);
                    acc_333[8] = z_461.x;
                    acc_333[9] = z_461.y;
                    float2 _f2_314 = make_float2(ssx, ssy);
                    float2 ss2_462 = fma_f32x2_rn_noftz(z_461, z_461, _f2_314);
                    ssx = ss2_462.x;
                    ssy = ss2_462.y;
                    float2 _f2_315 = make_float2(acc_333[10], acc_333[11]);
                    float2 a2_463 = _f2_315;
                    float2 _f2_316 = make_float2(__uint_as_float(t_447[5] << 16), __uint_as_float(t_447[5] & 4294901760u));
                    float2 z_464 = add_f32x2_noftz(a2_463, _f2_316);
                    acc_333[10] = z_464.x;
                    acc_333[11] = z_464.y;
                    float2 _f2_317 = make_float2(ssx, ssy);
                    float2 ss2_465 = fma_f32x2_rn_noftz(z_464, z_464, _f2_317);
                    ssx = ss2_465.x;
                    ssy = ss2_465.y;
                    float2 _f2_318 = make_float2(acc_333[12], acc_333[13]);
                    float2 a2_466 = _f2_318;
                    float2 _f2_319 = make_float2(__uint_as_float(t_447[6] << 16), __uint_as_float(t_447[6] & 4294901760u));
                    float2 z_467 = add_f32x2_noftz(a2_466, _f2_319);
                    acc_333[12] = z_467.x;
                    acc_333[13] = z_467.y;
                    float2 _f2_320 = make_float2(ssx, ssy);
                    float2 ss2_468 = fma_f32x2_rn_noftz(z_467, z_467, _f2_320);
                    ssx = ss2_468.x;
                    ssy = ss2_468.y;
                    float2 _f2_321 = make_float2(acc_333[14], acc_333[15]);
                    float2 a2_469 = _f2_321;
                    float2 _f2_322 = make_float2(__uint_as_float(t_447[7] << 16), __uint_as_float(t_447[7] & 4294901760u));
                    float2 z_470 = add_f32x2_noftz(a2_469, _f2_322);
                    acc_333[14] = z_470.x;
                    acc_333[15] = z_470.y;
                    float2 _f2_323 = make_float2(ssx, ssy);
                    float2 ss2_471 = fma_f32x2_rn_noftz(z_470, z_470, _f2_323);
                    ssx = ss2_471.x;
                    ssy = ss2_471.y;
                    float2 _f2_324 = make_float2(acc_333[16], acc_333[17]);
                    float2 a2_472 = _f2_324;
                    float2 _f2_325 = make_float2(__uint_as_float(t_447[8] << 16), __uint_as_float(t_447[8] & 4294901760u));
                    float2 z_473 = add_f32x2_noftz(a2_472, _f2_325);
                    acc_333[16] = z_473.x;
                    acc_333[17] = z_473.y;
                    float2 _f2_326 = make_float2(ssx, ssy);
                    float2 ss2_474 = fma_f32x2_rn_noftz(z_473, z_473, _f2_326);
                    ssx = ss2_474.x;
                    ssy = ss2_474.y;
                    float2 _f2_327 = make_float2(acc_333[18], acc_333[19]);
                    float2 a2_475 = _f2_327;
                    float2 _f2_328 = make_float2(__uint_as_float(t_447[9] << 16), __uint_as_float(t_447[9] & 4294901760u));
                    float2 z_476 = add_f32x2_noftz(a2_475, _f2_328);
                    acc_333[18] = z_476.x;
                    acc_333[19] = z_476.y;
                    float2 _f2_329 = make_float2(ssx, ssy);
                    float2 ss2_477 = fma_f32x2_rn_noftz(z_476, z_476, _f2_329);
                    ssx = ss2_477.x;
                    ssy = ss2_477.y;
                    float2 _f2_330 = make_float2(acc_333[20], acc_333[21]);
                    float2 a2_478 = _f2_330;
                    float2 _f2_331 = make_float2(__uint_as_float(t_447[10] << 16), __uint_as_float(t_447[10] & 4294901760u));
                    float2 z_479 = add_f32x2_noftz(a2_478, _f2_331);
                    acc_333[20] = z_479.x;
                    acc_333[21] = z_479.y;
                    float2 _f2_332 = make_float2(ssx, ssy);
                    float2 ss2_480 = fma_f32x2_rn_noftz(z_479, z_479, _f2_332);
                    ssx = ss2_480.x;
                    ssy = ss2_480.y;
                    float2 _f2_333 = make_float2(acc_333[22], acc_333[23]);
                    float2 a2_481 = _f2_333;
                    float2 _f2_334 = make_float2(__uint_as_float(t_447[11] << 16), __uint_as_float(t_447[11] & 4294901760u));
                    float2 z_482 = add_f32x2_noftz(a2_481, _f2_334);
                    acc_333[22] = z_482.x;
                    acc_333[23] = z_482.y;
                    float2 _f2_335 = make_float2(ssx, ssy);
                    float2 ss2_483 = fma_f32x2_rn_noftz(z_482, z_482, _f2_335);
                    ssx = ss2_483.x;
                    ssy = ss2_483.y;
                    float2 _f2_336 = make_float2(acc_333[24], acc_333[25]);
                    float2 a2_484 = _f2_336;
                    float2 _f2_337 = make_float2(__uint_as_float(t_447[12] << 16), __uint_as_float(t_447[12] & 4294901760u));
                    float2 z_485 = add_f32x2_noftz(a2_484, _f2_337);
                    acc_333[24] = z_485.x;
                    acc_333[25] = z_485.y;
                    float2 _f2_338 = make_float2(ssx, ssy);
                    float2 ss2_486 = fma_f32x2_rn_noftz(z_485, z_485, _f2_338);
                    ssx = ss2_486.x;
                    ssy = ss2_486.y;
                    float2 _f2_339 = make_float2(acc_333[26], acc_333[27]);
                    float2 a2_487 = _f2_339;
                    float2 _f2_340 = make_float2(__uint_as_float(t_447[13] << 16), __uint_as_float(t_447[13] & 4294901760u));
                    float2 z_488 = add_f32x2_noftz(a2_487, _f2_340);
                    acc_333[26] = z_488.x;
                    acc_333[27] = z_488.y;
                    float2 _f2_341 = make_float2(ssx, ssy);
                    float2 ss2_489 = fma_f32x2_rn_noftz(z_488, z_488, _f2_341);
                    ssx = ss2_489.x;
                    ssy = ss2_489.y;
                    float2 _f2_342 = make_float2(acc_333[28], acc_333[29]);
                    float2 a2_490 = _f2_342;
                    float2 _f2_343 = make_float2(__uint_as_float(t_447[14] << 16), __uint_as_float(t_447[14] & 4294901760u));
                    float2 z_491 = add_f32x2_noftz(a2_490, _f2_343);
                    acc_333[28] = z_491.x;
                    acc_333[29] = z_491.y;
                    float2 _f2_344 = make_float2(ssx, ssy);
                    float2 ss2_492 = fma_f32x2_rn_noftz(z_491, z_491, _f2_344);
                    ssx = ss2_492.x;
                    ssy = ss2_492.y;
                    float2 _f2_345 = make_float2(acc_333[30], acc_333[31]);
                    float2 a2_493 = _f2_345;
                    float2 _f2_346 = make_float2(__uint_as_float(t_447[15] << 16), __uint_as_float(t_447[15] & 4294901760u));
                    float2 z_494 = add_f32x2_noftz(a2_493, _f2_346);
                    acc_333[30] = z_494.x;
                    acc_333[31] = z_494.y;
                    float2 _f2_347 = make_float2(ssx, ssy);
                    float2 ss2_495 = fma_f32x2_rn_noftz(z_494, z_494, _f2_347);
                    ssx = ss2_495.x;
                    ssy = ss2_495.y;
                    unsigned int acc_333_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_333[_lp*2 + 0], acc_333[_lp*2+1 + 0]));
                        acc_333_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 0))), "r"(acc_333_bf16[0]), "r"(acc_333_bf16[1]), "r"(acc_333_bf16[2]), "r"(acc_333_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 1))), "r"(acc_333_bf16[4]), "r"(acc_333_bf16[5]), "r"(acc_333_bf16[6]), "r"(acc_333_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 2))), "r"(acc_333_bf16[8]), "r"(acc_333_bf16[9]), "r"(acc_333_bf16[10]), "r"(acc_333_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_326 + 16 * (f ^ 3))), "r"(acc_333_bf16[12]), "r"(acc_333_bf16[13]), "r"(acc_333_bf16[14]), "r"(acc_333_bf16[15]) : "memory");
                    int pp_496 = lane_2 + 96;
                    unsigned int own_497[4];
                    unsigned int own_sc_498[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_497[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_497[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_497[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_497[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_496)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_498[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_496)));
                    unsigned int a_499[4];
                    unsigned int b_500[4];
                    a_499[0] = ((first) ? own_497[0] : pv[12]);
                    b_500[0] = ((first) ? pv[12] : own_497[0]);
                    a_499[1] = ((first) ? own_497[1] : pv[13]);
                    b_500[1] = ((first) ? pv[13] : own_497[1]);
                    a_499[2] = ((first) ? own_497[2] : pv[14]);
                    b_500[2] = ((first) ? pv[14] : own_497[2]);
                    a_499[3] = ((first) ? own_497[3] : pv[15]);
                    b_500[3] = ((first) ? pv[15] : own_497[3]);
                    unsigned int a_sc_501 = ((first) ? own_sc_498[0] : ps[3]);
                    unsigned int b_sc_502 = ((first) ? ps[3] : own_sc_498[0]);
                    float acc_503[32] = {0};
                    bool fhi_504 = (f & 2) != 0;
                    bool fodd_505 = (f & 1) != 0;
                    bool fhi_506 = (f & 2) != 0;
                    unsigned int s_507[4];
                    s_507[0] = a_499[0];
                    s_507[1] = a_499[1];
                    s_507[2] = a_499[2];
                    s_507[3] = a_499[3];
                    if (fodd_505) {
                        s_507[0] = a_499[1];
                        s_507[1] = a_499[0];
                        s_507[2] = a_499[3];
                        s_507[3] = a_499[2];
                    }
                    unsigned int r_508[4];
                    r_508[0] = s_507[0];
                    r_508[1] = s_507[1];
                    r_508[2] = s_507[2];
                    r_508[3] = s_507[3];
                    if (fhi_506) {
                        r_508[0] = s_507[2];
                        r_508[1] = s_507[3];
                        r_508[2] = s_507[0];
                        r_508[3] = s_507[1];
                    }
                    bool fodd_509 = (f & 1) != 0;
                    bool fhi_510 = (f & 2) != 0;
                    unsigned int s_511[4];
                    s_511[0] = b_500[0];
                    s_511[1] = b_500[1];
                    s_511[2] = b_500[2];
                    s_511[3] = b_500[3];
                    if (fodd_509) {
                        s_511[0] = b_500[1];
                        s_511[1] = b_500[0];
                        s_511[2] = b_500[3];
                        s_511[3] = b_500[2];
                    }
                    unsigned int r_512[4];
                    r_512[0] = s_511[0];
                    r_512[1] = s_511[1];
                    r_512[2] = s_511[2];
                    r_512[3] = s_511[3];
                    if (fhi_510) {
                        r_512[0] = s_511[2];
                        r_512[1] = s_511[3];
                        r_512[2] = s_511[0];
                        r_512[3] = s_511[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_6) : "h"((uint16_t)(a_sc_501)));
                    float _cvt_f32_f16_12;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(_e4m3x2_to_f16x2_6)));
                    float _cvt_f32_f16_13;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(_e4m3x2_to_f16x2_6 >> 16)));
                    float _mul_12;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_cvt_f32_f16_12), "f"(rs_a));
                    float _mul_13;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_cvt_f32_f16_13), "f"(rs_a));
                    float lo_513 = _mul_12;
                    float hi_514 = _mul_13;
                    if (fhi_504) {
                        lo_513 = _mul_13;
                        hi_514 = _mul_12;
                    }
                    float2 _f2_348 = make_float2(lo_513, lo_513);
                    float2 scale0_515 = _f2_348;
                    float2 _f2_349 = make_float2(hi_514, hi_514);
                    float2 scale1_516 = _f2_349;
                    float _unpack_fp4_24[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_508[0]));
                        (&_unpack_fp4_24[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_24[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_24[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_24[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_24[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_24[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_24[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_24[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_24[0]), "=f"(_unpack_fp4_24[1]), "=f"(_unpack_fp4_24[2]), "=f"(_unpack_fp4_24[3]), "=f"(_unpack_fp4_24[4]), "=f"(_unpack_fp4_24[5]), "=f"(_unpack_fp4_24[6]), "=f"(_unpack_fp4_24[7]) : "r"(r_508[0]));
                        #endif
                    }
                    float2 _f2_350 = make_float2(acc_503[0], acc_503[1]);
                    float2 a_517 = _f2_350;
                    float2 _f2_351 = make_float2(_unpack_fp4_24[0], _unpack_fp4_24[1]);
                    float2 q_518 = _f2_351;
                    float2 r_519 = fma_f32x2_rn_noftz(q_518, scale0_515, a_517);
                    acc_503[0] = r_519.x;
                    acc_503[1] = r_519.y;
                    float2 _f2_352 = make_float2(acc_503[2], acc_503[3]);
                    float2 a_520 = _f2_352;
                    float2 _f2_353 = make_float2(_unpack_fp4_24[2], _unpack_fp4_24[3]);
                    float2 q_521 = _f2_353;
                    float2 r_522 = fma_f32x2_rn_noftz(q_521, scale0_515, a_520);
                    acc_503[2] = r_522.x;
                    acc_503[3] = r_522.y;
                    float2 _f2_354 = make_float2(acc_503[4], acc_503[5]);
                    float2 a_523 = _f2_354;
                    float2 _f2_355 = make_float2(_unpack_fp4_24[4], _unpack_fp4_24[5]);
                    float2 q_524 = _f2_355;
                    float2 r_525 = fma_f32x2_rn_noftz(q_524, scale0_515, a_523);
                    acc_503[4] = r_525.x;
                    acc_503[5] = r_525.y;
                    float2 _f2_356 = make_float2(acc_503[6], acc_503[7]);
                    float2 a_526 = _f2_356;
                    float2 _f2_357 = make_float2(_unpack_fp4_24[6], _unpack_fp4_24[7]);
                    float2 q_527 = _f2_357;
                    float2 r_528 = fma_f32x2_rn_noftz(q_527, scale0_515, a_526);
                    acc_503[6] = r_528.x;
                    acc_503[7] = r_528.y;
                    float _unpack_fp4_25[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_508[1 + 0]));
                        (&_unpack_fp4_25[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_25[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_25[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_25[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_25[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_25[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_25[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_25[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_25[0]), "=f"(_unpack_fp4_25[1]), "=f"(_unpack_fp4_25[2]), "=f"(_unpack_fp4_25[3]), "=f"(_unpack_fp4_25[4]), "=f"(_unpack_fp4_25[5]), "=f"(_unpack_fp4_25[6]), "=f"(_unpack_fp4_25[7]) : "r"(r_508[1 + 0]));
                        #endif
                    }
                    float2 _f2_358 = make_float2(acc_503[8], acc_503[9]);
                    float2 a_529 = _f2_358;
                    float2 _f2_359 = make_float2(_unpack_fp4_25[0], _unpack_fp4_25[1]);
                    float2 q_530 = _f2_359;
                    float2 r_531 = fma_f32x2_rn_noftz(q_530, scale0_515, a_529);
                    acc_503[8] = r_531.x;
                    acc_503[9] = r_531.y;
                    float2 _f2_360 = make_float2(acc_503[10], acc_503[11]);
                    float2 a_532 = _f2_360;
                    float2 _f2_361 = make_float2(_unpack_fp4_25[2], _unpack_fp4_25[3]);
                    float2 q_533 = _f2_361;
                    float2 r_534 = fma_f32x2_rn_noftz(q_533, scale0_515, a_532);
                    acc_503[10] = r_534.x;
                    acc_503[11] = r_534.y;
                    float2 _f2_362 = make_float2(acc_503[12], acc_503[13]);
                    float2 a_535 = _f2_362;
                    float2 _f2_363 = make_float2(_unpack_fp4_25[4], _unpack_fp4_25[5]);
                    float2 q_536 = _f2_363;
                    float2 r_537 = fma_f32x2_rn_noftz(q_536, scale0_515, a_535);
                    acc_503[12] = r_537.x;
                    acc_503[13] = r_537.y;
                    float2 _f2_364 = make_float2(acc_503[14], acc_503[15]);
                    float2 a_538 = _f2_364;
                    float2 _f2_365 = make_float2(_unpack_fp4_25[6], _unpack_fp4_25[7]);
                    float2 q_539 = _f2_365;
                    float2 r_540 = fma_f32x2_rn_noftz(q_539, scale0_515, a_538);
                    acc_503[14] = r_540.x;
                    acc_503[15] = r_540.y;
                    float _unpack_fp4_26[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_508[2 + 0]));
                        (&_unpack_fp4_26[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_26[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_26[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_26[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_26[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_26[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_26[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_26[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_26[0]), "=f"(_unpack_fp4_26[1]), "=f"(_unpack_fp4_26[2]), "=f"(_unpack_fp4_26[3]), "=f"(_unpack_fp4_26[4]), "=f"(_unpack_fp4_26[5]), "=f"(_unpack_fp4_26[6]), "=f"(_unpack_fp4_26[7]) : "r"(r_508[2 + 0]));
                        #endif
                    }
                    float2 _f2_366 = make_float2(acc_503[16], acc_503[17]);
                    float2 a_541 = _f2_366;
                    float2 _f2_367 = make_float2(_unpack_fp4_26[0], _unpack_fp4_26[1]);
                    float2 q_542 = _f2_367;
                    float2 r_543 = fma_f32x2_rn_noftz(q_542, scale1_516, a_541);
                    acc_503[16] = r_543.x;
                    acc_503[17] = r_543.y;
                    float2 _f2_368 = make_float2(acc_503[18], acc_503[19]);
                    float2 a_544 = _f2_368;
                    float2 _f2_369 = make_float2(_unpack_fp4_26[2], _unpack_fp4_26[3]);
                    float2 q_545 = _f2_369;
                    float2 r_546 = fma_f32x2_rn_noftz(q_545, scale1_516, a_544);
                    acc_503[18] = r_546.x;
                    acc_503[19] = r_546.y;
                    float2 _f2_370 = make_float2(acc_503[20], acc_503[21]);
                    float2 a_547 = _f2_370;
                    float2 _f2_371 = make_float2(_unpack_fp4_26[4], _unpack_fp4_26[5]);
                    float2 q_548 = _f2_371;
                    float2 r_549 = fma_f32x2_rn_noftz(q_548, scale1_516, a_547);
                    acc_503[20] = r_549.x;
                    acc_503[21] = r_549.y;
                    float2 _f2_372 = make_float2(acc_503[22], acc_503[23]);
                    float2 a_550 = _f2_372;
                    float2 _f2_373 = make_float2(_unpack_fp4_26[6], _unpack_fp4_26[7]);
                    float2 q_551 = _f2_373;
                    float2 r_552 = fma_f32x2_rn_noftz(q_551, scale1_516, a_550);
                    acc_503[22] = r_552.x;
                    acc_503[23] = r_552.y;
                    float _unpack_fp4_27[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_508[3 + 0]));
                        (&_unpack_fp4_27[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_27[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_27[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_27[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_27[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_27[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_27[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_27[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_27[0]), "=f"(_unpack_fp4_27[1]), "=f"(_unpack_fp4_27[2]), "=f"(_unpack_fp4_27[3]), "=f"(_unpack_fp4_27[4]), "=f"(_unpack_fp4_27[5]), "=f"(_unpack_fp4_27[6]), "=f"(_unpack_fp4_27[7]) : "r"(r_508[3 + 0]));
                        #endif
                    }
                    float2 _f2_374 = make_float2(acc_503[24], acc_503[25]);
                    float2 a_553 = _f2_374;
                    float2 _f2_375 = make_float2(_unpack_fp4_27[0], _unpack_fp4_27[1]);
                    float2 q_554 = _f2_375;
                    float2 r_555 = fma_f32x2_rn_noftz(q_554, scale1_516, a_553);
                    acc_503[24] = r_555.x;
                    acc_503[25] = r_555.y;
                    float2 _f2_376 = make_float2(acc_503[26], acc_503[27]);
                    float2 a_556 = _f2_376;
                    float2 _f2_377 = make_float2(_unpack_fp4_27[2], _unpack_fp4_27[3]);
                    float2 q_557 = _f2_377;
                    float2 r_558 = fma_f32x2_rn_noftz(q_557, scale1_516, a_556);
                    acc_503[26] = r_558.x;
                    acc_503[27] = r_558.y;
                    float2 _f2_378 = make_float2(acc_503[28], acc_503[29]);
                    float2 a_559 = _f2_378;
                    float2 _f2_379 = make_float2(_unpack_fp4_27[4], _unpack_fp4_27[5]);
                    float2 q_560 = _f2_379;
                    float2 r_561 = fma_f32x2_rn_noftz(q_560, scale1_516, a_559);
                    acc_503[28] = r_561.x;
                    acc_503[29] = r_561.y;
                    float2 _f2_380 = make_float2(acc_503[30], acc_503[31]);
                    float2 a_562 = _f2_380;
                    float2 _f2_381 = make_float2(_unpack_fp4_27[6], _unpack_fp4_27[7]);
                    float2 q_563 = _f2_381;
                    float2 r_564 = fma_f32x2_rn_noftz(q_563, scale1_516, a_562);
                    acc_503[30] = r_564.x;
                    acc_503[31] = r_564.y;
                    uint32_t _e4m3x2_to_f16x2_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_7) : "h"((uint16_t)(b_sc_502)));
                    float _cvt_f32_f16_14;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(_e4m3x2_to_f16x2_7)));
                    float _cvt_f32_f16_15;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(_e4m3x2_to_f16x2_7 >> 16)));
                    float _mul_14;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_cvt_f32_f16_14), "f"(rs_b));
                    float _mul_15;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_cvt_f32_f16_15), "f"(rs_b));
                    float lo_565 = _mul_14;
                    float hi_566 = _mul_15;
                    if (fhi_504) {
                        lo_565 = _mul_15;
                        hi_566 = _mul_14;
                    }
                    float2 _f2_382 = make_float2(lo_565, lo_565);
                    float2 scale0_567 = _f2_382;
                    float2 _f2_383 = make_float2(hi_566, hi_566);
                    float2 scale1_568 = _f2_383;
                    float _unpack_fp4_28[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_512[0]));
                        (&_unpack_fp4_28[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_28[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_28[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_28[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_28[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_28[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_28[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_28[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_28[0]), "=f"(_unpack_fp4_28[1]), "=f"(_unpack_fp4_28[2]), "=f"(_unpack_fp4_28[3]), "=f"(_unpack_fp4_28[4]), "=f"(_unpack_fp4_28[5]), "=f"(_unpack_fp4_28[6]), "=f"(_unpack_fp4_28[7]) : "r"(r_512[0]));
                        #endif
                    }
                    float2 _f2_384 = make_float2(acc_503[0], acc_503[1]);
                    float2 a_569 = _f2_384;
                    float2 _f2_385 = make_float2(_unpack_fp4_28[0], _unpack_fp4_28[1]);
                    float2 q_570 = _f2_385;
                    float2 r_571 = fma_f32x2_rn_noftz(q_570, scale0_567, a_569);
                    acc_503[0] = r_571.x;
                    acc_503[1] = r_571.y;
                    float2 _f2_386 = make_float2(acc_503[2], acc_503[3]);
                    float2 a_572 = _f2_386;
                    float2 _f2_387 = make_float2(_unpack_fp4_28[2], _unpack_fp4_28[3]);
                    float2 q_573 = _f2_387;
                    float2 r_574 = fma_f32x2_rn_noftz(q_573, scale0_567, a_572);
                    acc_503[2] = r_574.x;
                    acc_503[3] = r_574.y;
                    float2 _f2_388 = make_float2(acc_503[4], acc_503[5]);
                    float2 a_575 = _f2_388;
                    float2 _f2_389 = make_float2(_unpack_fp4_28[4], _unpack_fp4_28[5]);
                    float2 q_576 = _f2_389;
                    float2 r_577 = fma_f32x2_rn_noftz(q_576, scale0_567, a_575);
                    acc_503[4] = r_577.x;
                    acc_503[5] = r_577.y;
                    float2 _f2_390 = make_float2(acc_503[6], acc_503[7]);
                    float2 a_578 = _f2_390;
                    float2 _f2_391 = make_float2(_unpack_fp4_28[6], _unpack_fp4_28[7]);
                    float2 q_579 = _f2_391;
                    float2 r_580 = fma_f32x2_rn_noftz(q_579, scale0_567, a_578);
                    acc_503[6] = r_580.x;
                    acc_503[7] = r_580.y;
                    float _unpack_fp4_29[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_512[1 + 0]));
                        (&_unpack_fp4_29[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_29[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_29[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_29[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_29[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_29[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_29[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_29[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_29[0]), "=f"(_unpack_fp4_29[1]), "=f"(_unpack_fp4_29[2]), "=f"(_unpack_fp4_29[3]), "=f"(_unpack_fp4_29[4]), "=f"(_unpack_fp4_29[5]), "=f"(_unpack_fp4_29[6]), "=f"(_unpack_fp4_29[7]) : "r"(r_512[1 + 0]));
                        #endif
                    }
                    float2 _f2_392 = make_float2(acc_503[8], acc_503[9]);
                    float2 a_581 = _f2_392;
                    float2 _f2_393 = make_float2(_unpack_fp4_29[0], _unpack_fp4_29[1]);
                    float2 q_582 = _f2_393;
                    float2 r_583 = fma_f32x2_rn_noftz(q_582, scale0_567, a_581);
                    acc_503[8] = r_583.x;
                    acc_503[9] = r_583.y;
                    float2 _f2_394 = make_float2(acc_503[10], acc_503[11]);
                    float2 a_584 = _f2_394;
                    float2 _f2_395 = make_float2(_unpack_fp4_29[2], _unpack_fp4_29[3]);
                    float2 q_585 = _f2_395;
                    float2 r_586 = fma_f32x2_rn_noftz(q_585, scale0_567, a_584);
                    acc_503[10] = r_586.x;
                    acc_503[11] = r_586.y;
                    float2 _f2_396 = make_float2(acc_503[12], acc_503[13]);
                    float2 a_587 = _f2_396;
                    float2 _f2_397 = make_float2(_unpack_fp4_29[4], _unpack_fp4_29[5]);
                    float2 q_588 = _f2_397;
                    float2 r_589 = fma_f32x2_rn_noftz(q_588, scale0_567, a_587);
                    acc_503[12] = r_589.x;
                    acc_503[13] = r_589.y;
                    float2 _f2_398 = make_float2(acc_503[14], acc_503[15]);
                    float2 a_590 = _f2_398;
                    float2 _f2_399 = make_float2(_unpack_fp4_29[6], _unpack_fp4_29[7]);
                    float2 q_591 = _f2_399;
                    float2 r_592 = fma_f32x2_rn_noftz(q_591, scale0_567, a_590);
                    acc_503[14] = r_592.x;
                    acc_503[15] = r_592.y;
                    float _unpack_fp4_30[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_512[2 + 0]));
                        (&_unpack_fp4_30[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_30[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_30[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_30[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_30[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_30[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_30[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_30[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_30[0]), "=f"(_unpack_fp4_30[1]), "=f"(_unpack_fp4_30[2]), "=f"(_unpack_fp4_30[3]), "=f"(_unpack_fp4_30[4]), "=f"(_unpack_fp4_30[5]), "=f"(_unpack_fp4_30[6]), "=f"(_unpack_fp4_30[7]) : "r"(r_512[2 + 0]));
                        #endif
                    }
                    float2 _f2_400 = make_float2(acc_503[16], acc_503[17]);
                    float2 a_593 = _f2_400;
                    float2 _f2_401 = make_float2(_unpack_fp4_30[0], _unpack_fp4_30[1]);
                    float2 q_594 = _f2_401;
                    float2 r_595 = fma_f32x2_rn_noftz(q_594, scale1_568, a_593);
                    acc_503[16] = r_595.x;
                    acc_503[17] = r_595.y;
                    float2 _f2_402 = make_float2(acc_503[18], acc_503[19]);
                    float2 a_596 = _f2_402;
                    float2 _f2_403 = make_float2(_unpack_fp4_30[2], _unpack_fp4_30[3]);
                    float2 q_597 = _f2_403;
                    float2 r_598 = fma_f32x2_rn_noftz(q_597, scale1_568, a_596);
                    acc_503[18] = r_598.x;
                    acc_503[19] = r_598.y;
                    float2 _f2_404 = make_float2(acc_503[20], acc_503[21]);
                    float2 a_599 = _f2_404;
                    float2 _f2_405 = make_float2(_unpack_fp4_30[4], _unpack_fp4_30[5]);
                    float2 q_600 = _f2_405;
                    float2 r_601 = fma_f32x2_rn_noftz(q_600, scale1_568, a_599);
                    acc_503[20] = r_601.x;
                    acc_503[21] = r_601.y;
                    float2 _f2_406 = make_float2(acc_503[22], acc_503[23]);
                    float2 a_602 = _f2_406;
                    float2 _f2_407 = make_float2(_unpack_fp4_30[6], _unpack_fp4_30[7]);
                    float2 q_603 = _f2_407;
                    float2 r_604 = fma_f32x2_rn_noftz(q_603, scale1_568, a_602);
                    acc_503[22] = r_604.x;
                    acc_503[23] = r_604.y;
                    float _unpack_fp4_31[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_512[3 + 0]));
                        (&_unpack_fp4_31[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_31[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_31[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_31[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_31[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_31[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_31[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_31[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_31[0]), "=f"(_unpack_fp4_31[1]), "=f"(_unpack_fp4_31[2]), "=f"(_unpack_fp4_31[3]), "=f"(_unpack_fp4_31[4]), "=f"(_unpack_fp4_31[5]), "=f"(_unpack_fp4_31[6]), "=f"(_unpack_fp4_31[7]) : "r"(r_512[3 + 0]));
                        #endif
                    }
                    float2 _f2_408 = make_float2(acc_503[24], acc_503[25]);
                    float2 a_605 = _f2_408;
                    float2 _f2_409 = make_float2(_unpack_fp4_31[0], _unpack_fp4_31[1]);
                    float2 q_606 = _f2_409;
                    float2 r_607 = fma_f32x2_rn_noftz(q_606, scale1_568, a_605);
                    acc_503[24] = r_607.x;
                    acc_503[25] = r_607.y;
                    float2 _f2_410 = make_float2(acc_503[26], acc_503[27]);
                    float2 a_608 = _f2_410;
                    float2 _f2_411 = make_float2(_unpack_fp4_31[2], _unpack_fp4_31[3]);
                    float2 q_609 = _f2_411;
                    float2 r_610 = fma_f32x2_rn_noftz(q_609, scale1_568, a_608);
                    acc_503[26] = r_610.x;
                    acc_503[27] = r_610.y;
                    float2 _f2_412 = make_float2(acc_503[28], acc_503[29]);
                    float2 a_611 = _f2_412;
                    float2 _f2_413 = make_float2(_unpack_fp4_31[4], _unpack_fp4_31[5]);
                    float2 q_612 = _f2_413;
                    float2 r_613 = fma_f32x2_rn_noftz(q_612, scale1_568, a_611);
                    acc_503[28] = r_613.x;
                    acc_503[29] = r_613.y;
                    float2 _f2_414 = make_float2(acc_503[30], acc_503[31]);
                    float2 a_614 = _f2_414;
                    float2 _f2_415 = make_float2(_unpack_fp4_31[6], _unpack_fp4_31[7]);
                    float2 q_615 = _f2_415;
                    float2 r_616 = fma_f32x2_rn_noftz(q_615, scale1_568, a_614);
                    acc_503[30] = r_616.x;
                    acc_503[31] = r_616.y;
                    unsigned int t_617[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_617[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_617[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_617[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_617[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_617[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 3))));
                    float2 _f2_416 = make_float2(acc_503[0], acc_503[1]);
                    float2 a2_618 = _f2_416;
                    float2 _f2_417 = make_float2(__uint_as_float(t_617[0] << 16), __uint_as_float(t_617[0] & 4294901760u));
                    float2 z_619 = add_f32x2_noftz(a2_618, _f2_417);
                    acc_503[0] = z_619.x;
                    acc_503[1] = z_619.y;
                    float2 _f2_418 = make_float2(ssx, ssy);
                    float2 ss2_620 = fma_f32x2_rn_noftz(z_619, z_619, _f2_418);
                    ssx = ss2_620.x;
                    ssy = ss2_620.y;
                    float2 _f2_419 = make_float2(acc_503[2], acc_503[3]);
                    float2 a2_621 = _f2_419;
                    float2 _f2_420 = make_float2(__uint_as_float(t_617[1] << 16), __uint_as_float(t_617[1] & 4294901760u));
                    float2 z_622 = add_f32x2_noftz(a2_621, _f2_420);
                    acc_503[2] = z_622.x;
                    acc_503[3] = z_622.y;
                    float2 _f2_421 = make_float2(ssx, ssy);
                    float2 ss2_623 = fma_f32x2_rn_noftz(z_622, z_622, _f2_421);
                    ssx = ss2_623.x;
                    ssy = ss2_623.y;
                    float2 _f2_422 = make_float2(acc_503[4], acc_503[5]);
                    float2 a2_624 = _f2_422;
                    float2 _f2_423 = make_float2(__uint_as_float(t_617[2] << 16), __uint_as_float(t_617[2] & 4294901760u));
                    float2 z_625 = add_f32x2_noftz(a2_624, _f2_423);
                    acc_503[4] = z_625.x;
                    acc_503[5] = z_625.y;
                    float2 _f2_424 = make_float2(ssx, ssy);
                    float2 ss2_626 = fma_f32x2_rn_noftz(z_625, z_625, _f2_424);
                    ssx = ss2_626.x;
                    ssy = ss2_626.y;
                    float2 _f2_425 = make_float2(acc_503[6], acc_503[7]);
                    float2 a2_627 = _f2_425;
                    float2 _f2_426 = make_float2(__uint_as_float(t_617[3] << 16), __uint_as_float(t_617[3] & 4294901760u));
                    float2 z_628 = add_f32x2_noftz(a2_627, _f2_426);
                    acc_503[6] = z_628.x;
                    acc_503[7] = z_628.y;
                    float2 _f2_427 = make_float2(ssx, ssy);
                    float2 ss2_629 = fma_f32x2_rn_noftz(z_628, z_628, _f2_427);
                    ssx = ss2_629.x;
                    ssy = ss2_629.y;
                    float2 _f2_428 = make_float2(acc_503[8], acc_503[9]);
                    float2 a2_630 = _f2_428;
                    float2 _f2_429 = make_float2(__uint_as_float(t_617[4] << 16), __uint_as_float(t_617[4] & 4294901760u));
                    float2 z_631 = add_f32x2_noftz(a2_630, _f2_429);
                    acc_503[8] = z_631.x;
                    acc_503[9] = z_631.y;
                    float2 _f2_430 = make_float2(ssx, ssy);
                    float2 ss2_632 = fma_f32x2_rn_noftz(z_631, z_631, _f2_430);
                    ssx = ss2_632.x;
                    ssy = ss2_632.y;
                    float2 _f2_431 = make_float2(acc_503[10], acc_503[11]);
                    float2 a2_633 = _f2_431;
                    float2 _f2_432 = make_float2(__uint_as_float(t_617[5] << 16), __uint_as_float(t_617[5] & 4294901760u));
                    float2 z_634 = add_f32x2_noftz(a2_633, _f2_432);
                    acc_503[10] = z_634.x;
                    acc_503[11] = z_634.y;
                    float2 _f2_433 = make_float2(ssx, ssy);
                    float2 ss2_635 = fma_f32x2_rn_noftz(z_634, z_634, _f2_433);
                    ssx = ss2_635.x;
                    ssy = ss2_635.y;
                    float2 _f2_434 = make_float2(acc_503[12], acc_503[13]);
                    float2 a2_636 = _f2_434;
                    float2 _f2_435 = make_float2(__uint_as_float(t_617[6] << 16), __uint_as_float(t_617[6] & 4294901760u));
                    float2 z_637 = add_f32x2_noftz(a2_636, _f2_435);
                    acc_503[12] = z_637.x;
                    acc_503[13] = z_637.y;
                    float2 _f2_436 = make_float2(ssx, ssy);
                    float2 ss2_638 = fma_f32x2_rn_noftz(z_637, z_637, _f2_436);
                    ssx = ss2_638.x;
                    ssy = ss2_638.y;
                    float2 _f2_437 = make_float2(acc_503[14], acc_503[15]);
                    float2 a2_639 = _f2_437;
                    float2 _f2_438 = make_float2(__uint_as_float(t_617[7] << 16), __uint_as_float(t_617[7] & 4294901760u));
                    float2 z_640 = add_f32x2_noftz(a2_639, _f2_438);
                    acc_503[14] = z_640.x;
                    acc_503[15] = z_640.y;
                    float2 _f2_439 = make_float2(ssx, ssy);
                    float2 ss2_641 = fma_f32x2_rn_noftz(z_640, z_640, _f2_439);
                    ssx = ss2_641.x;
                    ssy = ss2_641.y;
                    float2 _f2_440 = make_float2(acc_503[16], acc_503[17]);
                    float2 a2_642 = _f2_440;
                    float2 _f2_441 = make_float2(__uint_as_float(t_617[8] << 16), __uint_as_float(t_617[8] & 4294901760u));
                    float2 z_643 = add_f32x2_noftz(a2_642, _f2_441);
                    acc_503[16] = z_643.x;
                    acc_503[17] = z_643.y;
                    float2 _f2_442 = make_float2(ssx, ssy);
                    float2 ss2_644 = fma_f32x2_rn_noftz(z_643, z_643, _f2_442);
                    ssx = ss2_644.x;
                    ssy = ss2_644.y;
                    float2 _f2_443 = make_float2(acc_503[18], acc_503[19]);
                    float2 a2_645 = _f2_443;
                    float2 _f2_444 = make_float2(__uint_as_float(t_617[9] << 16), __uint_as_float(t_617[9] & 4294901760u));
                    float2 z_646 = add_f32x2_noftz(a2_645, _f2_444);
                    acc_503[18] = z_646.x;
                    acc_503[19] = z_646.y;
                    float2 _f2_445 = make_float2(ssx, ssy);
                    float2 ss2_647 = fma_f32x2_rn_noftz(z_646, z_646, _f2_445);
                    ssx = ss2_647.x;
                    ssy = ss2_647.y;
                    float2 _f2_446 = make_float2(acc_503[20], acc_503[21]);
                    float2 a2_648 = _f2_446;
                    float2 _f2_447 = make_float2(__uint_as_float(t_617[10] << 16), __uint_as_float(t_617[10] & 4294901760u));
                    float2 z_649 = add_f32x2_noftz(a2_648, _f2_447);
                    acc_503[20] = z_649.x;
                    acc_503[21] = z_649.y;
                    float2 _f2_448 = make_float2(ssx, ssy);
                    float2 ss2_650 = fma_f32x2_rn_noftz(z_649, z_649, _f2_448);
                    ssx = ss2_650.x;
                    ssy = ss2_650.y;
                    float2 _f2_449 = make_float2(acc_503[22], acc_503[23]);
                    float2 a2_651 = _f2_449;
                    float2 _f2_450 = make_float2(__uint_as_float(t_617[11] << 16), __uint_as_float(t_617[11] & 4294901760u));
                    float2 z_652 = add_f32x2_noftz(a2_651, _f2_450);
                    acc_503[22] = z_652.x;
                    acc_503[23] = z_652.y;
                    float2 _f2_451 = make_float2(ssx, ssy);
                    float2 ss2_653 = fma_f32x2_rn_noftz(z_652, z_652, _f2_451);
                    ssx = ss2_653.x;
                    ssy = ss2_653.y;
                    float2 _f2_452 = make_float2(acc_503[24], acc_503[25]);
                    float2 a2_654 = _f2_452;
                    float2 _f2_453 = make_float2(__uint_as_float(t_617[12] << 16), __uint_as_float(t_617[12] & 4294901760u));
                    float2 z_655 = add_f32x2_noftz(a2_654, _f2_453);
                    acc_503[24] = z_655.x;
                    acc_503[25] = z_655.y;
                    float2 _f2_454 = make_float2(ssx, ssy);
                    float2 ss2_656 = fma_f32x2_rn_noftz(z_655, z_655, _f2_454);
                    ssx = ss2_656.x;
                    ssy = ss2_656.y;
                    float2 _f2_455 = make_float2(acc_503[26], acc_503[27]);
                    float2 a2_657 = _f2_455;
                    float2 _f2_456 = make_float2(__uint_as_float(t_617[13] << 16), __uint_as_float(t_617[13] & 4294901760u));
                    float2 z_658 = add_f32x2_noftz(a2_657, _f2_456);
                    acc_503[26] = z_658.x;
                    acc_503[27] = z_658.y;
                    float2 _f2_457 = make_float2(ssx, ssy);
                    float2 ss2_659 = fma_f32x2_rn_noftz(z_658, z_658, _f2_457);
                    ssx = ss2_659.x;
                    ssy = ss2_659.y;
                    float2 _f2_458 = make_float2(acc_503[28], acc_503[29]);
                    float2 a2_660 = _f2_458;
                    float2 _f2_459 = make_float2(__uint_as_float(t_617[14] << 16), __uint_as_float(t_617[14] & 4294901760u));
                    float2 z_661 = add_f32x2_noftz(a2_660, _f2_459);
                    acc_503[28] = z_661.x;
                    acc_503[29] = z_661.y;
                    float2 _f2_460 = make_float2(ssx, ssy);
                    float2 ss2_662 = fma_f32x2_rn_noftz(z_661, z_661, _f2_460);
                    ssx = ss2_662.x;
                    ssy = ss2_662.y;
                    float2 _f2_461 = make_float2(acc_503[30], acc_503[31]);
                    float2 a2_663 = _f2_461;
                    float2 _f2_462 = make_float2(__uint_as_float(t_617[15] << 16), __uint_as_float(t_617[15] & 4294901760u));
                    float2 z_664 = add_f32x2_noftz(a2_663, _f2_462);
                    acc_503[30] = z_664.x;
                    acc_503[31] = z_664.y;
                    float2 _f2_463 = make_float2(ssx, ssy);
                    float2 ss2_665 = fma_f32x2_rn_noftz(z_664, z_664, _f2_463);
                    ssx = ss2_665.x;
                    ssy = ss2_665.y;
                    unsigned int acc_503_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_503[_lp*2 + 0], acc_503[_lp*2+1 + 0]));
                        acc_503_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 0))), "r"(acc_503_bf16[0]), "r"(acc_503_bf16[1]), "r"(acc_503_bf16[2]), "r"(acc_503_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 1))), "r"(acc_503_bf16[4]), "r"(acc_503_bf16[5]), "r"(acc_503_bf16[6]), "r"(acc_503_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 2))), "r"(acc_503_bf16[8]), "r"(acc_503_bf16[9]), "r"(acc_503_bf16[10]), "r"(acc_503_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_496 + 16 * (f ^ 3))), "r"(acc_503_bf16[12]), "r"(acc_503_bf16[13]), "r"(acc_503_bf16[14]), "r"(acc_503_bf16[15]) : "memory");
                    unsigned int d_666 = 0;
                    d_666 = d_666 | pv[16] & ~((pv[16] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[17] & ~((pv[17] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[18] & ~((pv[18] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[19] & ~((pv[19] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | ((ps[4] & 255) + 1 | (ps[4] >> 8) + 1) & 256;
                    d_666 = d_666 | pv[20] & ~((pv[20] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[21] & ~((pv[21] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[22] & ~((pv[22] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | pv[23] & ~((pv[23] & 2004318071) + 2004318071) & 2290649224u;
                    d_666 = d_666 | ((ps[5] & 255) + 1 | (ps[5] >> 8) + 1) & 256;
                    if (d_666 != 0) {
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
                    int pp_667 = lane_2 + 128;
                    unsigned int own_668[4];
                    unsigned int own_sc_669[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_668[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_668[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_668[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_668[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_667)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_669[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_667)));
                    unsigned int a_670[4];
                    unsigned int b_671[4];
                    a_670[0] = ((first) ? own_668[0] : pv[16]);
                    b_671[0] = ((first) ? pv[16] : own_668[0]);
                    a_670[1] = ((first) ? own_668[1] : pv[17]);
                    b_671[1] = ((first) ? pv[17] : own_668[1]);
                    a_670[2] = ((first) ? own_668[2] : pv[18]);
                    b_671[2] = ((first) ? pv[18] : own_668[2]);
                    a_670[3] = ((first) ? own_668[3] : pv[19]);
                    b_671[3] = ((first) ? pv[19] : own_668[3]);
                    unsigned int a_sc_672 = ((first) ? own_sc_669[0] : ps[4]);
                    unsigned int b_sc_673 = ((first) ? ps[4] : own_sc_669[0]);
                    float acc_674[32] = {0};
                    bool fhi_675 = (f & 2) != 0;
                    bool fodd_676 = (f & 1) != 0;
                    bool fhi_677 = (f & 2) != 0;
                    unsigned int s_678[4];
                    s_678[0] = a_670[0];
                    s_678[1] = a_670[1];
                    s_678[2] = a_670[2];
                    s_678[3] = a_670[3];
                    if (fodd_676) {
                        s_678[0] = a_670[1];
                        s_678[1] = a_670[0];
                        s_678[2] = a_670[3];
                        s_678[3] = a_670[2];
                    }
                    unsigned int r_679[4];
                    r_679[0] = s_678[0];
                    r_679[1] = s_678[1];
                    r_679[2] = s_678[2];
                    r_679[3] = s_678[3];
                    if (fhi_677) {
                        r_679[0] = s_678[2];
                        r_679[1] = s_678[3];
                        r_679[2] = s_678[0];
                        r_679[3] = s_678[1];
                    }
                    bool fodd_680 = (f & 1) != 0;
                    bool fhi_681 = (f & 2) != 0;
                    unsigned int s_682[4];
                    s_682[0] = b_671[0];
                    s_682[1] = b_671[1];
                    s_682[2] = b_671[2];
                    s_682[3] = b_671[3];
                    if (fodd_680) {
                        s_682[0] = b_671[1];
                        s_682[1] = b_671[0];
                        s_682[2] = b_671[3];
                        s_682[3] = b_671[2];
                    }
                    unsigned int r_683[4];
                    r_683[0] = s_682[0];
                    r_683[1] = s_682[1];
                    r_683[2] = s_682[2];
                    r_683[3] = s_682[3];
                    if (fhi_681) {
                        r_683[0] = s_682[2];
                        r_683[1] = s_682[3];
                        r_683[2] = s_682[0];
                        r_683[3] = s_682[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_8;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_8) : "h"((uint16_t)(a_sc_672)));
                    float _cvt_f32_f16_16;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(_e4m3x2_to_f16x2_8)));
                    float _cvt_f32_f16_17;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(_e4m3x2_to_f16x2_8 >> 16)));
                    float _mul_16;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_cvt_f32_f16_16), "f"(rs_a));
                    float _mul_17;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_cvt_f32_f16_17), "f"(rs_a));
                    float lo_684 = _mul_16;
                    float hi_685 = _mul_17;
                    if (fhi_675) {
                        lo_684 = _mul_17;
                        hi_685 = _mul_16;
                    }
                    float2 _f2_464 = make_float2(lo_684, lo_684);
                    float2 scale0_686 = _f2_464;
                    float2 _f2_465 = make_float2(hi_685, hi_685);
                    float2 scale1_687 = _f2_465;
                    float _unpack_fp4_32[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_679[0]));
                        (&_unpack_fp4_32[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_32[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_32[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_32[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_32[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_32[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_32[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_32[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_32[0]), "=f"(_unpack_fp4_32[1]), "=f"(_unpack_fp4_32[2]), "=f"(_unpack_fp4_32[3]), "=f"(_unpack_fp4_32[4]), "=f"(_unpack_fp4_32[5]), "=f"(_unpack_fp4_32[6]), "=f"(_unpack_fp4_32[7]) : "r"(r_679[0]));
                        #endif
                    }
                    float2 _f2_466 = make_float2(acc_674[0], acc_674[1]);
                    float2 a_688 = _f2_466;
                    float2 _f2_467 = make_float2(_unpack_fp4_32[0], _unpack_fp4_32[1]);
                    float2 q_689 = _f2_467;
                    float2 r_690 = fma_f32x2_rn_noftz(q_689, scale0_686, a_688);
                    acc_674[0] = r_690.x;
                    acc_674[1] = r_690.y;
                    float2 _f2_468 = make_float2(acc_674[2], acc_674[3]);
                    float2 a_691 = _f2_468;
                    float2 _f2_469 = make_float2(_unpack_fp4_32[2], _unpack_fp4_32[3]);
                    float2 q_692 = _f2_469;
                    float2 r_693 = fma_f32x2_rn_noftz(q_692, scale0_686, a_691);
                    acc_674[2] = r_693.x;
                    acc_674[3] = r_693.y;
                    float2 _f2_470 = make_float2(acc_674[4], acc_674[5]);
                    float2 a_694 = _f2_470;
                    float2 _f2_471 = make_float2(_unpack_fp4_32[4], _unpack_fp4_32[5]);
                    float2 q_695 = _f2_471;
                    float2 r_696 = fma_f32x2_rn_noftz(q_695, scale0_686, a_694);
                    acc_674[4] = r_696.x;
                    acc_674[5] = r_696.y;
                    float2 _f2_472 = make_float2(acc_674[6], acc_674[7]);
                    float2 a_697 = _f2_472;
                    float2 _f2_473 = make_float2(_unpack_fp4_32[6], _unpack_fp4_32[7]);
                    float2 q_698 = _f2_473;
                    float2 r_699 = fma_f32x2_rn_noftz(q_698, scale0_686, a_697);
                    acc_674[6] = r_699.x;
                    acc_674[7] = r_699.y;
                    float _unpack_fp4_33[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_679[1 + 0]));
                        (&_unpack_fp4_33[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_33[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_33[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_33[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_33[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_33[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_33[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_33[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_33[0]), "=f"(_unpack_fp4_33[1]), "=f"(_unpack_fp4_33[2]), "=f"(_unpack_fp4_33[3]), "=f"(_unpack_fp4_33[4]), "=f"(_unpack_fp4_33[5]), "=f"(_unpack_fp4_33[6]), "=f"(_unpack_fp4_33[7]) : "r"(r_679[1 + 0]));
                        #endif
                    }
                    float2 _f2_474 = make_float2(acc_674[8], acc_674[9]);
                    float2 a_700 = _f2_474;
                    float2 _f2_475 = make_float2(_unpack_fp4_33[0], _unpack_fp4_33[1]);
                    float2 q_701 = _f2_475;
                    float2 r_702 = fma_f32x2_rn_noftz(q_701, scale0_686, a_700);
                    acc_674[8] = r_702.x;
                    acc_674[9] = r_702.y;
                    float2 _f2_476 = make_float2(acc_674[10], acc_674[11]);
                    float2 a_703 = _f2_476;
                    float2 _f2_477 = make_float2(_unpack_fp4_33[2], _unpack_fp4_33[3]);
                    float2 q_704 = _f2_477;
                    float2 r_705 = fma_f32x2_rn_noftz(q_704, scale0_686, a_703);
                    acc_674[10] = r_705.x;
                    acc_674[11] = r_705.y;
                    float2 _f2_478 = make_float2(acc_674[12], acc_674[13]);
                    float2 a_706 = _f2_478;
                    float2 _f2_479 = make_float2(_unpack_fp4_33[4], _unpack_fp4_33[5]);
                    float2 q_707 = _f2_479;
                    float2 r_708 = fma_f32x2_rn_noftz(q_707, scale0_686, a_706);
                    acc_674[12] = r_708.x;
                    acc_674[13] = r_708.y;
                    float2 _f2_480 = make_float2(acc_674[14], acc_674[15]);
                    float2 a_709 = _f2_480;
                    float2 _f2_481 = make_float2(_unpack_fp4_33[6], _unpack_fp4_33[7]);
                    float2 q_710 = _f2_481;
                    float2 r_711 = fma_f32x2_rn_noftz(q_710, scale0_686, a_709);
                    acc_674[14] = r_711.x;
                    acc_674[15] = r_711.y;
                    float _unpack_fp4_34[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_679[2 + 0]));
                        (&_unpack_fp4_34[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_34[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_34[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_34[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_34[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_34[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_34[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_34[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_34[0]), "=f"(_unpack_fp4_34[1]), "=f"(_unpack_fp4_34[2]), "=f"(_unpack_fp4_34[3]), "=f"(_unpack_fp4_34[4]), "=f"(_unpack_fp4_34[5]), "=f"(_unpack_fp4_34[6]), "=f"(_unpack_fp4_34[7]) : "r"(r_679[2 + 0]));
                        #endif
                    }
                    float2 _f2_482 = make_float2(acc_674[16], acc_674[17]);
                    float2 a_712 = _f2_482;
                    float2 _f2_483 = make_float2(_unpack_fp4_34[0], _unpack_fp4_34[1]);
                    float2 q_713 = _f2_483;
                    float2 r_714 = fma_f32x2_rn_noftz(q_713, scale1_687, a_712);
                    acc_674[16] = r_714.x;
                    acc_674[17] = r_714.y;
                    float2 _f2_484 = make_float2(acc_674[18], acc_674[19]);
                    float2 a_715 = _f2_484;
                    float2 _f2_485 = make_float2(_unpack_fp4_34[2], _unpack_fp4_34[3]);
                    float2 q_716 = _f2_485;
                    float2 r_717 = fma_f32x2_rn_noftz(q_716, scale1_687, a_715);
                    acc_674[18] = r_717.x;
                    acc_674[19] = r_717.y;
                    float2 _f2_486 = make_float2(acc_674[20], acc_674[21]);
                    float2 a_718 = _f2_486;
                    float2 _f2_487 = make_float2(_unpack_fp4_34[4], _unpack_fp4_34[5]);
                    float2 q_719 = _f2_487;
                    float2 r_720 = fma_f32x2_rn_noftz(q_719, scale1_687, a_718);
                    acc_674[20] = r_720.x;
                    acc_674[21] = r_720.y;
                    float2 _f2_488 = make_float2(acc_674[22], acc_674[23]);
                    float2 a_721 = _f2_488;
                    float2 _f2_489 = make_float2(_unpack_fp4_34[6], _unpack_fp4_34[7]);
                    float2 q_722 = _f2_489;
                    float2 r_723 = fma_f32x2_rn_noftz(q_722, scale1_687, a_721);
                    acc_674[22] = r_723.x;
                    acc_674[23] = r_723.y;
                    float _unpack_fp4_35[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_679[3 + 0]));
                        (&_unpack_fp4_35[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_35[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_35[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_35[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_35[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_35[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_35[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_35[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_35[0]), "=f"(_unpack_fp4_35[1]), "=f"(_unpack_fp4_35[2]), "=f"(_unpack_fp4_35[3]), "=f"(_unpack_fp4_35[4]), "=f"(_unpack_fp4_35[5]), "=f"(_unpack_fp4_35[6]), "=f"(_unpack_fp4_35[7]) : "r"(r_679[3 + 0]));
                        #endif
                    }
                    float2 _f2_490 = make_float2(acc_674[24], acc_674[25]);
                    float2 a_724 = _f2_490;
                    float2 _f2_491 = make_float2(_unpack_fp4_35[0], _unpack_fp4_35[1]);
                    float2 q_725 = _f2_491;
                    float2 r_726 = fma_f32x2_rn_noftz(q_725, scale1_687, a_724);
                    acc_674[24] = r_726.x;
                    acc_674[25] = r_726.y;
                    float2 _f2_492 = make_float2(acc_674[26], acc_674[27]);
                    float2 a_727 = _f2_492;
                    float2 _f2_493 = make_float2(_unpack_fp4_35[2], _unpack_fp4_35[3]);
                    float2 q_728 = _f2_493;
                    float2 r_729 = fma_f32x2_rn_noftz(q_728, scale1_687, a_727);
                    acc_674[26] = r_729.x;
                    acc_674[27] = r_729.y;
                    float2 _f2_494 = make_float2(acc_674[28], acc_674[29]);
                    float2 a_730 = _f2_494;
                    float2 _f2_495 = make_float2(_unpack_fp4_35[4], _unpack_fp4_35[5]);
                    float2 q_731 = _f2_495;
                    float2 r_732 = fma_f32x2_rn_noftz(q_731, scale1_687, a_730);
                    acc_674[28] = r_732.x;
                    acc_674[29] = r_732.y;
                    float2 _f2_496 = make_float2(acc_674[30], acc_674[31]);
                    float2 a_733 = _f2_496;
                    float2 _f2_497 = make_float2(_unpack_fp4_35[6], _unpack_fp4_35[7]);
                    float2 q_734 = _f2_497;
                    float2 r_735 = fma_f32x2_rn_noftz(q_734, scale1_687, a_733);
                    acc_674[30] = r_735.x;
                    acc_674[31] = r_735.y;
                    uint32_t _e4m3x2_to_f16x2_9;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_9) : "h"((uint16_t)(b_sc_673)));
                    float _cvt_f32_f16_18;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(_e4m3x2_to_f16x2_9)));
                    float _cvt_f32_f16_19;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(_e4m3x2_to_f16x2_9 >> 16)));
                    float _mul_18;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_cvt_f32_f16_18), "f"(rs_b));
                    float _mul_19;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_cvt_f32_f16_19), "f"(rs_b));
                    float lo_736 = _mul_18;
                    float hi_737 = _mul_19;
                    if (fhi_675) {
                        lo_736 = _mul_19;
                        hi_737 = _mul_18;
                    }
                    float2 _f2_498 = make_float2(lo_736, lo_736);
                    float2 scale0_738 = _f2_498;
                    float2 _f2_499 = make_float2(hi_737, hi_737);
                    float2 scale1_739 = _f2_499;
                    float _unpack_fp4_36[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_683[0]));
                        (&_unpack_fp4_36[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_36[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_36[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_36[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_36[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_36[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_36[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_36[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_36[0]), "=f"(_unpack_fp4_36[1]), "=f"(_unpack_fp4_36[2]), "=f"(_unpack_fp4_36[3]), "=f"(_unpack_fp4_36[4]), "=f"(_unpack_fp4_36[5]), "=f"(_unpack_fp4_36[6]), "=f"(_unpack_fp4_36[7]) : "r"(r_683[0]));
                        #endif
                    }
                    float2 _f2_500 = make_float2(acc_674[0], acc_674[1]);
                    float2 a_740 = _f2_500;
                    float2 _f2_501 = make_float2(_unpack_fp4_36[0], _unpack_fp4_36[1]);
                    float2 q_741 = _f2_501;
                    float2 r_742 = fma_f32x2_rn_noftz(q_741, scale0_738, a_740);
                    acc_674[0] = r_742.x;
                    acc_674[1] = r_742.y;
                    float2 _f2_502 = make_float2(acc_674[2], acc_674[3]);
                    float2 a_743 = _f2_502;
                    float2 _f2_503 = make_float2(_unpack_fp4_36[2], _unpack_fp4_36[3]);
                    float2 q_744 = _f2_503;
                    float2 r_745 = fma_f32x2_rn_noftz(q_744, scale0_738, a_743);
                    acc_674[2] = r_745.x;
                    acc_674[3] = r_745.y;
                    float2 _f2_504 = make_float2(acc_674[4], acc_674[5]);
                    float2 a_746 = _f2_504;
                    float2 _f2_505 = make_float2(_unpack_fp4_36[4], _unpack_fp4_36[5]);
                    float2 q_747 = _f2_505;
                    float2 r_748 = fma_f32x2_rn_noftz(q_747, scale0_738, a_746);
                    acc_674[4] = r_748.x;
                    acc_674[5] = r_748.y;
                    float2 _f2_506 = make_float2(acc_674[6], acc_674[7]);
                    float2 a_749 = _f2_506;
                    float2 _f2_507 = make_float2(_unpack_fp4_36[6], _unpack_fp4_36[7]);
                    float2 q_750 = _f2_507;
                    float2 r_751 = fma_f32x2_rn_noftz(q_750, scale0_738, a_749);
                    acc_674[6] = r_751.x;
                    acc_674[7] = r_751.y;
                    float _unpack_fp4_37[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_683[1 + 0]));
                        (&_unpack_fp4_37[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_37[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_37[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_37[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_37[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_37[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_37[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_37[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_37[0]), "=f"(_unpack_fp4_37[1]), "=f"(_unpack_fp4_37[2]), "=f"(_unpack_fp4_37[3]), "=f"(_unpack_fp4_37[4]), "=f"(_unpack_fp4_37[5]), "=f"(_unpack_fp4_37[6]), "=f"(_unpack_fp4_37[7]) : "r"(r_683[1 + 0]));
                        #endif
                    }
                    float2 _f2_508 = make_float2(acc_674[8], acc_674[9]);
                    float2 a_752 = _f2_508;
                    float2 _f2_509 = make_float2(_unpack_fp4_37[0], _unpack_fp4_37[1]);
                    float2 q_753 = _f2_509;
                    float2 r_754 = fma_f32x2_rn_noftz(q_753, scale0_738, a_752);
                    acc_674[8] = r_754.x;
                    acc_674[9] = r_754.y;
                    float2 _f2_510 = make_float2(acc_674[10], acc_674[11]);
                    float2 a_755 = _f2_510;
                    float2 _f2_511 = make_float2(_unpack_fp4_37[2], _unpack_fp4_37[3]);
                    float2 q_756 = _f2_511;
                    float2 r_757 = fma_f32x2_rn_noftz(q_756, scale0_738, a_755);
                    acc_674[10] = r_757.x;
                    acc_674[11] = r_757.y;
                    float2 _f2_512 = make_float2(acc_674[12], acc_674[13]);
                    float2 a_758 = _f2_512;
                    float2 _f2_513 = make_float2(_unpack_fp4_37[4], _unpack_fp4_37[5]);
                    float2 q_759 = _f2_513;
                    float2 r_760 = fma_f32x2_rn_noftz(q_759, scale0_738, a_758);
                    acc_674[12] = r_760.x;
                    acc_674[13] = r_760.y;
                    float2 _f2_514 = make_float2(acc_674[14], acc_674[15]);
                    float2 a_761 = _f2_514;
                    float2 _f2_515 = make_float2(_unpack_fp4_37[6], _unpack_fp4_37[7]);
                    float2 q_762 = _f2_515;
                    float2 r_763 = fma_f32x2_rn_noftz(q_762, scale0_738, a_761);
                    acc_674[14] = r_763.x;
                    acc_674[15] = r_763.y;
                    float _unpack_fp4_38[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_683[2 + 0]));
                        (&_unpack_fp4_38[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_38[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_38[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_38[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_38[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_38[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_38[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_38[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_38[0]), "=f"(_unpack_fp4_38[1]), "=f"(_unpack_fp4_38[2]), "=f"(_unpack_fp4_38[3]), "=f"(_unpack_fp4_38[4]), "=f"(_unpack_fp4_38[5]), "=f"(_unpack_fp4_38[6]), "=f"(_unpack_fp4_38[7]) : "r"(r_683[2 + 0]));
                        #endif
                    }
                    float2 _f2_516 = make_float2(acc_674[16], acc_674[17]);
                    float2 a_764 = _f2_516;
                    float2 _f2_517 = make_float2(_unpack_fp4_38[0], _unpack_fp4_38[1]);
                    float2 q_765 = _f2_517;
                    float2 r_766 = fma_f32x2_rn_noftz(q_765, scale1_739, a_764);
                    acc_674[16] = r_766.x;
                    acc_674[17] = r_766.y;
                    float2 _f2_518 = make_float2(acc_674[18], acc_674[19]);
                    float2 a_767 = _f2_518;
                    float2 _f2_519 = make_float2(_unpack_fp4_38[2], _unpack_fp4_38[3]);
                    float2 q_768 = _f2_519;
                    float2 r_769 = fma_f32x2_rn_noftz(q_768, scale1_739, a_767);
                    acc_674[18] = r_769.x;
                    acc_674[19] = r_769.y;
                    float2 _f2_520 = make_float2(acc_674[20], acc_674[21]);
                    float2 a_770 = _f2_520;
                    float2 _f2_521 = make_float2(_unpack_fp4_38[4], _unpack_fp4_38[5]);
                    float2 q_771 = _f2_521;
                    float2 r_772 = fma_f32x2_rn_noftz(q_771, scale1_739, a_770);
                    acc_674[20] = r_772.x;
                    acc_674[21] = r_772.y;
                    float2 _f2_522 = make_float2(acc_674[22], acc_674[23]);
                    float2 a_773 = _f2_522;
                    float2 _f2_523 = make_float2(_unpack_fp4_38[6], _unpack_fp4_38[7]);
                    float2 q_774 = _f2_523;
                    float2 r_775 = fma_f32x2_rn_noftz(q_774, scale1_739, a_773);
                    acc_674[22] = r_775.x;
                    acc_674[23] = r_775.y;
                    float _unpack_fp4_39[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_683[3 + 0]));
                        (&_unpack_fp4_39[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_39[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_39[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_39[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_39[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_39[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_39[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_39[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_39[0]), "=f"(_unpack_fp4_39[1]), "=f"(_unpack_fp4_39[2]), "=f"(_unpack_fp4_39[3]), "=f"(_unpack_fp4_39[4]), "=f"(_unpack_fp4_39[5]), "=f"(_unpack_fp4_39[6]), "=f"(_unpack_fp4_39[7]) : "r"(r_683[3 + 0]));
                        #endif
                    }
                    float2 _f2_524 = make_float2(acc_674[24], acc_674[25]);
                    float2 a_776 = _f2_524;
                    float2 _f2_525 = make_float2(_unpack_fp4_39[0], _unpack_fp4_39[1]);
                    float2 q_777 = _f2_525;
                    float2 r_778 = fma_f32x2_rn_noftz(q_777, scale1_739, a_776);
                    acc_674[24] = r_778.x;
                    acc_674[25] = r_778.y;
                    float2 _f2_526 = make_float2(acc_674[26], acc_674[27]);
                    float2 a_779 = _f2_526;
                    float2 _f2_527 = make_float2(_unpack_fp4_39[2], _unpack_fp4_39[3]);
                    float2 q_780 = _f2_527;
                    float2 r_781 = fma_f32x2_rn_noftz(q_780, scale1_739, a_779);
                    acc_674[26] = r_781.x;
                    acc_674[27] = r_781.y;
                    float2 _f2_528 = make_float2(acc_674[28], acc_674[29]);
                    float2 a_782 = _f2_528;
                    float2 _f2_529 = make_float2(_unpack_fp4_39[4], _unpack_fp4_39[5]);
                    float2 q_783 = _f2_529;
                    float2 r_784 = fma_f32x2_rn_noftz(q_783, scale1_739, a_782);
                    acc_674[28] = r_784.x;
                    acc_674[29] = r_784.y;
                    float2 _f2_530 = make_float2(acc_674[30], acc_674[31]);
                    float2 a_785 = _f2_530;
                    float2 _f2_531 = make_float2(_unpack_fp4_39[6], _unpack_fp4_39[7]);
                    float2 q_786 = _f2_531;
                    float2 r_787 = fma_f32x2_rn_noftz(q_786, scale1_739, a_785);
                    acc_674[30] = r_787.x;
                    acc_674[31] = r_787.y;
                    unsigned int t_788[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_788[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_788[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_788[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_788[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_788[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 3))));
                    float2 _f2_532 = make_float2(acc_674[0], acc_674[1]);
                    float2 a2_789 = _f2_532;
                    float2 _f2_533 = make_float2(__uint_as_float(t_788[0] << 16), __uint_as_float(t_788[0] & 4294901760u));
                    float2 z_790 = add_f32x2_noftz(a2_789, _f2_533);
                    acc_674[0] = z_790.x;
                    acc_674[1] = z_790.y;
                    float2 _f2_534 = make_float2(ssx, ssy);
                    float2 ss2_791 = fma_f32x2_rn_noftz(z_790, z_790, _f2_534);
                    ssx = ss2_791.x;
                    ssy = ss2_791.y;
                    float2 _f2_535 = make_float2(acc_674[2], acc_674[3]);
                    float2 a2_792 = _f2_535;
                    float2 _f2_536 = make_float2(__uint_as_float(t_788[1] << 16), __uint_as_float(t_788[1] & 4294901760u));
                    float2 z_793 = add_f32x2_noftz(a2_792, _f2_536);
                    acc_674[2] = z_793.x;
                    acc_674[3] = z_793.y;
                    float2 _f2_537 = make_float2(ssx, ssy);
                    float2 ss2_794 = fma_f32x2_rn_noftz(z_793, z_793, _f2_537);
                    ssx = ss2_794.x;
                    ssy = ss2_794.y;
                    float2 _f2_538 = make_float2(acc_674[4], acc_674[5]);
                    float2 a2_795 = _f2_538;
                    float2 _f2_539 = make_float2(__uint_as_float(t_788[2] << 16), __uint_as_float(t_788[2] & 4294901760u));
                    float2 z_796 = add_f32x2_noftz(a2_795, _f2_539);
                    acc_674[4] = z_796.x;
                    acc_674[5] = z_796.y;
                    float2 _f2_540 = make_float2(ssx, ssy);
                    float2 ss2_797 = fma_f32x2_rn_noftz(z_796, z_796, _f2_540);
                    ssx = ss2_797.x;
                    ssy = ss2_797.y;
                    float2 _f2_541 = make_float2(acc_674[6], acc_674[7]);
                    float2 a2_798 = _f2_541;
                    float2 _f2_542 = make_float2(__uint_as_float(t_788[3] << 16), __uint_as_float(t_788[3] & 4294901760u));
                    float2 z_799 = add_f32x2_noftz(a2_798, _f2_542);
                    acc_674[6] = z_799.x;
                    acc_674[7] = z_799.y;
                    float2 _f2_543 = make_float2(ssx, ssy);
                    float2 ss2_800 = fma_f32x2_rn_noftz(z_799, z_799, _f2_543);
                    ssx = ss2_800.x;
                    ssy = ss2_800.y;
                    float2 _f2_544 = make_float2(acc_674[8], acc_674[9]);
                    float2 a2_801 = _f2_544;
                    float2 _f2_545 = make_float2(__uint_as_float(t_788[4] << 16), __uint_as_float(t_788[4] & 4294901760u));
                    float2 z_802 = add_f32x2_noftz(a2_801, _f2_545);
                    acc_674[8] = z_802.x;
                    acc_674[9] = z_802.y;
                    float2 _f2_546 = make_float2(ssx, ssy);
                    float2 ss2_803 = fma_f32x2_rn_noftz(z_802, z_802, _f2_546);
                    ssx = ss2_803.x;
                    ssy = ss2_803.y;
                    float2 _f2_547 = make_float2(acc_674[10], acc_674[11]);
                    float2 a2_804 = _f2_547;
                    float2 _f2_548 = make_float2(__uint_as_float(t_788[5] << 16), __uint_as_float(t_788[5] & 4294901760u));
                    float2 z_805 = add_f32x2_noftz(a2_804, _f2_548);
                    acc_674[10] = z_805.x;
                    acc_674[11] = z_805.y;
                    float2 _f2_549 = make_float2(ssx, ssy);
                    float2 ss2_806 = fma_f32x2_rn_noftz(z_805, z_805, _f2_549);
                    ssx = ss2_806.x;
                    ssy = ss2_806.y;
                    float2 _f2_550 = make_float2(acc_674[12], acc_674[13]);
                    float2 a2_807 = _f2_550;
                    float2 _f2_551 = make_float2(__uint_as_float(t_788[6] << 16), __uint_as_float(t_788[6] & 4294901760u));
                    float2 z_808 = add_f32x2_noftz(a2_807, _f2_551);
                    acc_674[12] = z_808.x;
                    acc_674[13] = z_808.y;
                    float2 _f2_552 = make_float2(ssx, ssy);
                    float2 ss2_809 = fma_f32x2_rn_noftz(z_808, z_808, _f2_552);
                    ssx = ss2_809.x;
                    ssy = ss2_809.y;
                    float2 _f2_553 = make_float2(acc_674[14], acc_674[15]);
                    float2 a2_810 = _f2_553;
                    float2 _f2_554 = make_float2(__uint_as_float(t_788[7] << 16), __uint_as_float(t_788[7] & 4294901760u));
                    float2 z_811 = add_f32x2_noftz(a2_810, _f2_554);
                    acc_674[14] = z_811.x;
                    acc_674[15] = z_811.y;
                    float2 _f2_555 = make_float2(ssx, ssy);
                    float2 ss2_812 = fma_f32x2_rn_noftz(z_811, z_811, _f2_555);
                    ssx = ss2_812.x;
                    ssy = ss2_812.y;
                    float2 _f2_556 = make_float2(acc_674[16], acc_674[17]);
                    float2 a2_813 = _f2_556;
                    float2 _f2_557 = make_float2(__uint_as_float(t_788[8] << 16), __uint_as_float(t_788[8] & 4294901760u));
                    float2 z_814 = add_f32x2_noftz(a2_813, _f2_557);
                    acc_674[16] = z_814.x;
                    acc_674[17] = z_814.y;
                    float2 _f2_558 = make_float2(ssx, ssy);
                    float2 ss2_815 = fma_f32x2_rn_noftz(z_814, z_814, _f2_558);
                    ssx = ss2_815.x;
                    ssy = ss2_815.y;
                    float2 _f2_559 = make_float2(acc_674[18], acc_674[19]);
                    float2 a2_816 = _f2_559;
                    float2 _f2_560 = make_float2(__uint_as_float(t_788[9] << 16), __uint_as_float(t_788[9] & 4294901760u));
                    float2 z_817 = add_f32x2_noftz(a2_816, _f2_560);
                    acc_674[18] = z_817.x;
                    acc_674[19] = z_817.y;
                    float2 _f2_561 = make_float2(ssx, ssy);
                    float2 ss2_818 = fma_f32x2_rn_noftz(z_817, z_817, _f2_561);
                    ssx = ss2_818.x;
                    ssy = ss2_818.y;
                    float2 _f2_562 = make_float2(acc_674[20], acc_674[21]);
                    float2 a2_819 = _f2_562;
                    float2 _f2_563 = make_float2(__uint_as_float(t_788[10] << 16), __uint_as_float(t_788[10] & 4294901760u));
                    float2 z_820 = add_f32x2_noftz(a2_819, _f2_563);
                    acc_674[20] = z_820.x;
                    acc_674[21] = z_820.y;
                    float2 _f2_564 = make_float2(ssx, ssy);
                    float2 ss2_821 = fma_f32x2_rn_noftz(z_820, z_820, _f2_564);
                    ssx = ss2_821.x;
                    ssy = ss2_821.y;
                    float2 _f2_565 = make_float2(acc_674[22], acc_674[23]);
                    float2 a2_822 = _f2_565;
                    float2 _f2_566 = make_float2(__uint_as_float(t_788[11] << 16), __uint_as_float(t_788[11] & 4294901760u));
                    float2 z_823 = add_f32x2_noftz(a2_822, _f2_566);
                    acc_674[22] = z_823.x;
                    acc_674[23] = z_823.y;
                    float2 _f2_567 = make_float2(ssx, ssy);
                    float2 ss2_824 = fma_f32x2_rn_noftz(z_823, z_823, _f2_567);
                    ssx = ss2_824.x;
                    ssy = ss2_824.y;
                    float2 _f2_568 = make_float2(acc_674[24], acc_674[25]);
                    float2 a2_825 = _f2_568;
                    float2 _f2_569 = make_float2(__uint_as_float(t_788[12] << 16), __uint_as_float(t_788[12] & 4294901760u));
                    float2 z_826 = add_f32x2_noftz(a2_825, _f2_569);
                    acc_674[24] = z_826.x;
                    acc_674[25] = z_826.y;
                    float2 _f2_570 = make_float2(ssx, ssy);
                    float2 ss2_827 = fma_f32x2_rn_noftz(z_826, z_826, _f2_570);
                    ssx = ss2_827.x;
                    ssy = ss2_827.y;
                    float2 _f2_571 = make_float2(acc_674[26], acc_674[27]);
                    float2 a2_828 = _f2_571;
                    float2 _f2_572 = make_float2(__uint_as_float(t_788[13] << 16), __uint_as_float(t_788[13] & 4294901760u));
                    float2 z_829 = add_f32x2_noftz(a2_828, _f2_572);
                    acc_674[26] = z_829.x;
                    acc_674[27] = z_829.y;
                    float2 _f2_573 = make_float2(ssx, ssy);
                    float2 ss2_830 = fma_f32x2_rn_noftz(z_829, z_829, _f2_573);
                    ssx = ss2_830.x;
                    ssy = ss2_830.y;
                    float2 _f2_574 = make_float2(acc_674[28], acc_674[29]);
                    float2 a2_831 = _f2_574;
                    float2 _f2_575 = make_float2(__uint_as_float(t_788[14] << 16), __uint_as_float(t_788[14] & 4294901760u));
                    float2 z_832 = add_f32x2_noftz(a2_831, _f2_575);
                    acc_674[28] = z_832.x;
                    acc_674[29] = z_832.y;
                    float2 _f2_576 = make_float2(ssx, ssy);
                    float2 ss2_833 = fma_f32x2_rn_noftz(z_832, z_832, _f2_576);
                    ssx = ss2_833.x;
                    ssy = ss2_833.y;
                    float2 _f2_577 = make_float2(acc_674[30], acc_674[31]);
                    float2 a2_834 = _f2_577;
                    float2 _f2_578 = make_float2(__uint_as_float(t_788[15] << 16), __uint_as_float(t_788[15] & 4294901760u));
                    float2 z_835 = add_f32x2_noftz(a2_834, _f2_578);
                    acc_674[30] = z_835.x;
                    acc_674[31] = z_835.y;
                    float2 _f2_579 = make_float2(ssx, ssy);
                    float2 ss2_836 = fma_f32x2_rn_noftz(z_835, z_835, _f2_579);
                    ssx = ss2_836.x;
                    ssy = ss2_836.y;
                    unsigned int acc_674_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_674[_lp*2 + 0], acc_674[_lp*2+1 + 0]));
                        acc_674_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 0))), "r"(acc_674_bf16[0]), "r"(acc_674_bf16[1]), "r"(acc_674_bf16[2]), "r"(acc_674_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 1))), "r"(acc_674_bf16[4]), "r"(acc_674_bf16[5]), "r"(acc_674_bf16[6]), "r"(acc_674_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 2))), "r"(acc_674_bf16[8]), "r"(acc_674_bf16[9]), "r"(acc_674_bf16[10]), "r"(acc_674_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_667 + 16 * (f ^ 3))), "r"(acc_674_bf16[12]), "r"(acc_674_bf16[13]), "r"(acc_674_bf16[14]), "r"(acc_674_bf16[15]) : "memory");
                    int pp_837 = lane_2 + 160;
                    unsigned int own_838[4];
                    unsigned int own_sc_839[1];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&own_838[0])), "=r"(*reinterpret_cast<uint32_t*>(&own_838[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&own_838[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&own_838[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(dob_off + 16 * pp_837)));
                    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&own_sc_839[0])) : "r"(wsm_addr + (unsigned int)(dob_off + 3072 + 2 * pp_837)));
                    unsigned int a_840[4];
                    unsigned int b_841[4];
                    a_840[0] = ((first) ? own_838[0] : pv[20]);
                    b_841[0] = ((first) ? pv[20] : own_838[0]);
                    a_840[1] = ((first) ? own_838[1] : pv[21]);
                    b_841[1] = ((first) ? pv[21] : own_838[1]);
                    a_840[2] = ((first) ? own_838[2] : pv[22]);
                    b_841[2] = ((first) ? pv[22] : own_838[2]);
                    a_840[3] = ((first) ? own_838[3] : pv[23]);
                    b_841[3] = ((first) ? pv[23] : own_838[3]);
                    unsigned int a_sc_842 = ((first) ? own_sc_839[0] : ps[5]);
                    unsigned int b_sc_843 = ((first) ? ps[5] : own_sc_839[0]);
                    float acc_844[32] = {0};
                    bool fhi_845 = (f & 2) != 0;
                    bool fodd_846 = (f & 1) != 0;
                    bool fhi_847 = (f & 2) != 0;
                    unsigned int s_848[4];
                    s_848[0] = a_840[0];
                    s_848[1] = a_840[1];
                    s_848[2] = a_840[2];
                    s_848[3] = a_840[3];
                    if (fodd_846) {
                        s_848[0] = a_840[1];
                        s_848[1] = a_840[0];
                        s_848[2] = a_840[3];
                        s_848[3] = a_840[2];
                    }
                    unsigned int r_849[4];
                    r_849[0] = s_848[0];
                    r_849[1] = s_848[1];
                    r_849[2] = s_848[2];
                    r_849[3] = s_848[3];
                    if (fhi_847) {
                        r_849[0] = s_848[2];
                        r_849[1] = s_848[3];
                        r_849[2] = s_848[0];
                        r_849[3] = s_848[1];
                    }
                    bool fodd_850 = (f & 1) != 0;
                    bool fhi_851 = (f & 2) != 0;
                    unsigned int s_852[4];
                    s_852[0] = b_841[0];
                    s_852[1] = b_841[1];
                    s_852[2] = b_841[2];
                    s_852[3] = b_841[3];
                    if (fodd_850) {
                        s_852[0] = b_841[1];
                        s_852[1] = b_841[0];
                        s_852[2] = b_841[3];
                        s_852[3] = b_841[2];
                    }
                    unsigned int r_853[4];
                    r_853[0] = s_852[0];
                    r_853[1] = s_852[1];
                    r_853[2] = s_852[2];
                    r_853[3] = s_852[3];
                    if (fhi_851) {
                        r_853[0] = s_852[2];
                        r_853[1] = s_852[3];
                        r_853[2] = s_852[0];
                        r_853[3] = s_852[1];
                    }
                    uint32_t _e4m3x2_to_f16x2_10;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_10) : "h"((uint16_t)(a_sc_842)));
                    float _cvt_f32_f16_20;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(_e4m3x2_to_f16x2_10)));
                    float _cvt_f32_f16_21;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(_e4m3x2_to_f16x2_10 >> 16)));
                    float _mul_20;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_cvt_f32_f16_20), "f"(rs_a));
                    float _mul_21;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_cvt_f32_f16_21), "f"(rs_a));
                    float lo_854 = _mul_20;
                    float hi_855 = _mul_21;
                    if (fhi_845) {
                        lo_854 = _mul_21;
                        hi_855 = _mul_20;
                    }
                    float2 _f2_580 = make_float2(lo_854, lo_854);
                    float2 scale0_856 = _f2_580;
                    float2 _f2_581 = make_float2(hi_855, hi_855);
                    float2 scale1_857 = _f2_581;
                    float _unpack_fp4_40[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_849[0]));
                        (&_unpack_fp4_40[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_40[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_40[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_40[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_40[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_40[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_40[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_40[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_40[0]), "=f"(_unpack_fp4_40[1]), "=f"(_unpack_fp4_40[2]), "=f"(_unpack_fp4_40[3]), "=f"(_unpack_fp4_40[4]), "=f"(_unpack_fp4_40[5]), "=f"(_unpack_fp4_40[6]), "=f"(_unpack_fp4_40[7]) : "r"(r_849[0]));
                        #endif
                    }
                    float2 _f2_582 = make_float2(acc_844[0], acc_844[1]);
                    float2 a_858 = _f2_582;
                    float2 _f2_583 = make_float2(_unpack_fp4_40[0], _unpack_fp4_40[1]);
                    float2 q_859 = _f2_583;
                    float2 r_860 = fma_f32x2_rn_noftz(q_859, scale0_856, a_858);
                    acc_844[0] = r_860.x;
                    acc_844[1] = r_860.y;
                    float2 _f2_584 = make_float2(acc_844[2], acc_844[3]);
                    float2 a_861 = _f2_584;
                    float2 _f2_585 = make_float2(_unpack_fp4_40[2], _unpack_fp4_40[3]);
                    float2 q_862 = _f2_585;
                    float2 r_863 = fma_f32x2_rn_noftz(q_862, scale0_856, a_861);
                    acc_844[2] = r_863.x;
                    acc_844[3] = r_863.y;
                    float2 _f2_586 = make_float2(acc_844[4], acc_844[5]);
                    float2 a_864 = _f2_586;
                    float2 _f2_587 = make_float2(_unpack_fp4_40[4], _unpack_fp4_40[5]);
                    float2 q_865 = _f2_587;
                    float2 r_866 = fma_f32x2_rn_noftz(q_865, scale0_856, a_864);
                    acc_844[4] = r_866.x;
                    acc_844[5] = r_866.y;
                    float2 _f2_588 = make_float2(acc_844[6], acc_844[7]);
                    float2 a_867 = _f2_588;
                    float2 _f2_589 = make_float2(_unpack_fp4_40[6], _unpack_fp4_40[7]);
                    float2 q_868 = _f2_589;
                    float2 r_869 = fma_f32x2_rn_noftz(q_868, scale0_856, a_867);
                    acc_844[6] = r_869.x;
                    acc_844[7] = r_869.y;
                    float _unpack_fp4_41[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_849[1 + 0]));
                        (&_unpack_fp4_41[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_41[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_41[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_41[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_41[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_41[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_41[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_41[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_41[0]), "=f"(_unpack_fp4_41[1]), "=f"(_unpack_fp4_41[2]), "=f"(_unpack_fp4_41[3]), "=f"(_unpack_fp4_41[4]), "=f"(_unpack_fp4_41[5]), "=f"(_unpack_fp4_41[6]), "=f"(_unpack_fp4_41[7]) : "r"(r_849[1 + 0]));
                        #endif
                    }
                    float2 _f2_590 = make_float2(acc_844[8], acc_844[9]);
                    float2 a_870 = _f2_590;
                    float2 _f2_591 = make_float2(_unpack_fp4_41[0], _unpack_fp4_41[1]);
                    float2 q_871 = _f2_591;
                    float2 r_872 = fma_f32x2_rn_noftz(q_871, scale0_856, a_870);
                    acc_844[8] = r_872.x;
                    acc_844[9] = r_872.y;
                    float2 _f2_592 = make_float2(acc_844[10], acc_844[11]);
                    float2 a_873 = _f2_592;
                    float2 _f2_593 = make_float2(_unpack_fp4_41[2], _unpack_fp4_41[3]);
                    float2 q_874 = _f2_593;
                    float2 r_875 = fma_f32x2_rn_noftz(q_874, scale0_856, a_873);
                    acc_844[10] = r_875.x;
                    acc_844[11] = r_875.y;
                    float2 _f2_594 = make_float2(acc_844[12], acc_844[13]);
                    float2 a_876 = _f2_594;
                    float2 _f2_595 = make_float2(_unpack_fp4_41[4], _unpack_fp4_41[5]);
                    float2 q_877 = _f2_595;
                    float2 r_878 = fma_f32x2_rn_noftz(q_877, scale0_856, a_876);
                    acc_844[12] = r_878.x;
                    acc_844[13] = r_878.y;
                    float2 _f2_596 = make_float2(acc_844[14], acc_844[15]);
                    float2 a_879 = _f2_596;
                    float2 _f2_597 = make_float2(_unpack_fp4_41[6], _unpack_fp4_41[7]);
                    float2 q_880 = _f2_597;
                    float2 r_881 = fma_f32x2_rn_noftz(q_880, scale0_856, a_879);
                    acc_844[14] = r_881.x;
                    acc_844[15] = r_881.y;
                    float _unpack_fp4_42[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_849[2 + 0]));
                        (&_unpack_fp4_42[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_42[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_42[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_42[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_42[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_42[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_42[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_42[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_42[0]), "=f"(_unpack_fp4_42[1]), "=f"(_unpack_fp4_42[2]), "=f"(_unpack_fp4_42[3]), "=f"(_unpack_fp4_42[4]), "=f"(_unpack_fp4_42[5]), "=f"(_unpack_fp4_42[6]), "=f"(_unpack_fp4_42[7]) : "r"(r_849[2 + 0]));
                        #endif
                    }
                    float2 _f2_598 = make_float2(acc_844[16], acc_844[17]);
                    float2 a_882 = _f2_598;
                    float2 _f2_599 = make_float2(_unpack_fp4_42[0], _unpack_fp4_42[1]);
                    float2 q_883 = _f2_599;
                    float2 r_884 = fma_f32x2_rn_noftz(q_883, scale1_857, a_882);
                    acc_844[16] = r_884.x;
                    acc_844[17] = r_884.y;
                    float2 _f2_600 = make_float2(acc_844[18], acc_844[19]);
                    float2 a_885 = _f2_600;
                    float2 _f2_601 = make_float2(_unpack_fp4_42[2], _unpack_fp4_42[3]);
                    float2 q_886 = _f2_601;
                    float2 r_887 = fma_f32x2_rn_noftz(q_886, scale1_857, a_885);
                    acc_844[18] = r_887.x;
                    acc_844[19] = r_887.y;
                    float2 _f2_602 = make_float2(acc_844[20], acc_844[21]);
                    float2 a_888 = _f2_602;
                    float2 _f2_603 = make_float2(_unpack_fp4_42[4], _unpack_fp4_42[5]);
                    float2 q_889 = _f2_603;
                    float2 r_890 = fma_f32x2_rn_noftz(q_889, scale1_857, a_888);
                    acc_844[20] = r_890.x;
                    acc_844[21] = r_890.y;
                    float2 _f2_604 = make_float2(acc_844[22], acc_844[23]);
                    float2 a_891 = _f2_604;
                    float2 _f2_605 = make_float2(_unpack_fp4_42[6], _unpack_fp4_42[7]);
                    float2 q_892 = _f2_605;
                    float2 r_893 = fma_f32x2_rn_noftz(q_892, scale1_857, a_891);
                    acc_844[22] = r_893.x;
                    acc_844[23] = r_893.y;
                    float _unpack_fp4_43[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_849[3 + 0]));
                        (&_unpack_fp4_43[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_43[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_43[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_43[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_43[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_43[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_43[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_43[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_43[0]), "=f"(_unpack_fp4_43[1]), "=f"(_unpack_fp4_43[2]), "=f"(_unpack_fp4_43[3]), "=f"(_unpack_fp4_43[4]), "=f"(_unpack_fp4_43[5]), "=f"(_unpack_fp4_43[6]), "=f"(_unpack_fp4_43[7]) : "r"(r_849[3 + 0]));
                        #endif
                    }
                    float2 _f2_606 = make_float2(acc_844[24], acc_844[25]);
                    float2 a_894 = _f2_606;
                    float2 _f2_607 = make_float2(_unpack_fp4_43[0], _unpack_fp4_43[1]);
                    float2 q_895 = _f2_607;
                    float2 r_896 = fma_f32x2_rn_noftz(q_895, scale1_857, a_894);
                    acc_844[24] = r_896.x;
                    acc_844[25] = r_896.y;
                    float2 _f2_608 = make_float2(acc_844[26], acc_844[27]);
                    float2 a_897 = _f2_608;
                    float2 _f2_609 = make_float2(_unpack_fp4_43[2], _unpack_fp4_43[3]);
                    float2 q_898 = _f2_609;
                    float2 r_899 = fma_f32x2_rn_noftz(q_898, scale1_857, a_897);
                    acc_844[26] = r_899.x;
                    acc_844[27] = r_899.y;
                    float2 _f2_610 = make_float2(acc_844[28], acc_844[29]);
                    float2 a_900 = _f2_610;
                    float2 _f2_611 = make_float2(_unpack_fp4_43[4], _unpack_fp4_43[5]);
                    float2 q_901 = _f2_611;
                    float2 r_902 = fma_f32x2_rn_noftz(q_901, scale1_857, a_900);
                    acc_844[28] = r_902.x;
                    acc_844[29] = r_902.y;
                    float2 _f2_612 = make_float2(acc_844[30], acc_844[31]);
                    float2 a_903 = _f2_612;
                    float2 _f2_613 = make_float2(_unpack_fp4_43[6], _unpack_fp4_43[7]);
                    float2 q_904 = _f2_613;
                    float2 r_905 = fma_f32x2_rn_noftz(q_904, scale1_857, a_903);
                    acc_844[30] = r_905.x;
                    acc_844[31] = r_905.y;
                    uint32_t _e4m3x2_to_f16x2_11;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_11) : "h"((uint16_t)(b_sc_843)));
                    float _cvt_f32_f16_22;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(_e4m3x2_to_f16x2_11)));
                    float _cvt_f32_f16_23;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(_e4m3x2_to_f16x2_11 >> 16)));
                    float _mul_22;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_cvt_f32_f16_22), "f"(rs_b));
                    float _mul_23;
                    asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_cvt_f32_f16_23), "f"(rs_b));
                    float lo_906 = _mul_22;
                    float hi_907 = _mul_23;
                    if (fhi_845) {
                        lo_906 = _mul_23;
                        hi_907 = _mul_22;
                    }
                    float2 _f2_614 = make_float2(lo_906, lo_906);
                    float2 scale0_908 = _f2_614;
                    float2 _f2_615 = make_float2(hi_907, hi_907);
                    float2 scale1_909 = _f2_615;
                    float _unpack_fp4_44[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_853[0]));
                        (&_unpack_fp4_44[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_44[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_44[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_44[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_44[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_44[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_44[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_44[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_44[0]), "=f"(_unpack_fp4_44[1]), "=f"(_unpack_fp4_44[2]), "=f"(_unpack_fp4_44[3]), "=f"(_unpack_fp4_44[4]), "=f"(_unpack_fp4_44[5]), "=f"(_unpack_fp4_44[6]), "=f"(_unpack_fp4_44[7]) : "r"(r_853[0]));
                        #endif
                    }
                    float2 _f2_616 = make_float2(acc_844[0], acc_844[1]);
                    float2 a_910 = _f2_616;
                    float2 _f2_617 = make_float2(_unpack_fp4_44[0], _unpack_fp4_44[1]);
                    float2 q_911 = _f2_617;
                    float2 r_912 = fma_f32x2_rn_noftz(q_911, scale0_908, a_910);
                    acc_844[0] = r_912.x;
                    acc_844[1] = r_912.y;
                    float2 _f2_618 = make_float2(acc_844[2], acc_844[3]);
                    float2 a_913 = _f2_618;
                    float2 _f2_619 = make_float2(_unpack_fp4_44[2], _unpack_fp4_44[3]);
                    float2 q_914 = _f2_619;
                    float2 r_915 = fma_f32x2_rn_noftz(q_914, scale0_908, a_913);
                    acc_844[2] = r_915.x;
                    acc_844[3] = r_915.y;
                    float2 _f2_620 = make_float2(acc_844[4], acc_844[5]);
                    float2 a_916 = _f2_620;
                    float2 _f2_621 = make_float2(_unpack_fp4_44[4], _unpack_fp4_44[5]);
                    float2 q_917 = _f2_621;
                    float2 r_918 = fma_f32x2_rn_noftz(q_917, scale0_908, a_916);
                    acc_844[4] = r_918.x;
                    acc_844[5] = r_918.y;
                    float2 _f2_622 = make_float2(acc_844[6], acc_844[7]);
                    float2 a_919 = _f2_622;
                    float2 _f2_623 = make_float2(_unpack_fp4_44[6], _unpack_fp4_44[7]);
                    float2 q_920 = _f2_623;
                    float2 r_921 = fma_f32x2_rn_noftz(q_920, scale0_908, a_919);
                    acc_844[6] = r_921.x;
                    acc_844[7] = r_921.y;
                    float _unpack_fp4_45[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_853[1 + 0]));
                        (&_unpack_fp4_45[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_45[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_45[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_45[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_45[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_45[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_45[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_45[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_45[0]), "=f"(_unpack_fp4_45[1]), "=f"(_unpack_fp4_45[2]), "=f"(_unpack_fp4_45[3]), "=f"(_unpack_fp4_45[4]), "=f"(_unpack_fp4_45[5]), "=f"(_unpack_fp4_45[6]), "=f"(_unpack_fp4_45[7]) : "r"(r_853[1 + 0]));
                        #endif
                    }
                    float2 _f2_624 = make_float2(acc_844[8], acc_844[9]);
                    float2 a_922 = _f2_624;
                    float2 _f2_625 = make_float2(_unpack_fp4_45[0], _unpack_fp4_45[1]);
                    float2 q_923 = _f2_625;
                    float2 r_924 = fma_f32x2_rn_noftz(q_923, scale0_908, a_922);
                    acc_844[8] = r_924.x;
                    acc_844[9] = r_924.y;
                    float2 _f2_626 = make_float2(acc_844[10], acc_844[11]);
                    float2 a_925 = _f2_626;
                    float2 _f2_627 = make_float2(_unpack_fp4_45[2], _unpack_fp4_45[3]);
                    float2 q_926 = _f2_627;
                    float2 r_927 = fma_f32x2_rn_noftz(q_926, scale0_908, a_925);
                    acc_844[10] = r_927.x;
                    acc_844[11] = r_927.y;
                    float2 _f2_628 = make_float2(acc_844[12], acc_844[13]);
                    float2 a_928 = _f2_628;
                    float2 _f2_629 = make_float2(_unpack_fp4_45[4], _unpack_fp4_45[5]);
                    float2 q_929 = _f2_629;
                    float2 r_930 = fma_f32x2_rn_noftz(q_929, scale0_908, a_928);
                    acc_844[12] = r_930.x;
                    acc_844[13] = r_930.y;
                    float2 _f2_630 = make_float2(acc_844[14], acc_844[15]);
                    float2 a_931 = _f2_630;
                    float2 _f2_631 = make_float2(_unpack_fp4_45[6], _unpack_fp4_45[7]);
                    float2 q_932 = _f2_631;
                    float2 r_933 = fma_f32x2_rn_noftz(q_932, scale0_908, a_931);
                    acc_844[14] = r_933.x;
                    acc_844[15] = r_933.y;
                    float _unpack_fp4_46[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_853[2 + 0]));
                        (&_unpack_fp4_46[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_46[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_46[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_46[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_46[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_46[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_46[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_46[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_46[0]), "=f"(_unpack_fp4_46[1]), "=f"(_unpack_fp4_46[2]), "=f"(_unpack_fp4_46[3]), "=f"(_unpack_fp4_46[4]), "=f"(_unpack_fp4_46[5]), "=f"(_unpack_fp4_46[6]), "=f"(_unpack_fp4_46[7]) : "r"(r_853[2 + 0]));
                        #endif
                    }
                    float2 _f2_632 = make_float2(acc_844[16], acc_844[17]);
                    float2 a_934 = _f2_632;
                    float2 _f2_633 = make_float2(_unpack_fp4_46[0], _unpack_fp4_46[1]);
                    float2 q_935 = _f2_633;
                    float2 r_936 = fma_f32x2_rn_noftz(q_935, scale1_909, a_934);
                    acc_844[16] = r_936.x;
                    acc_844[17] = r_936.y;
                    float2 _f2_634 = make_float2(acc_844[18], acc_844[19]);
                    float2 a_937 = _f2_634;
                    float2 _f2_635 = make_float2(_unpack_fp4_46[2], _unpack_fp4_46[3]);
                    float2 q_938 = _f2_635;
                    float2 r_939 = fma_f32x2_rn_noftz(q_938, scale1_909, a_937);
                    acc_844[18] = r_939.x;
                    acc_844[19] = r_939.y;
                    float2 _f2_636 = make_float2(acc_844[20], acc_844[21]);
                    float2 a_940 = _f2_636;
                    float2 _f2_637 = make_float2(_unpack_fp4_46[4], _unpack_fp4_46[5]);
                    float2 q_941 = _f2_637;
                    float2 r_942 = fma_f32x2_rn_noftz(q_941, scale1_909, a_940);
                    acc_844[20] = r_942.x;
                    acc_844[21] = r_942.y;
                    float2 _f2_638 = make_float2(acc_844[22], acc_844[23]);
                    float2 a_943 = _f2_638;
                    float2 _f2_639 = make_float2(_unpack_fp4_46[6], _unpack_fp4_46[7]);
                    float2 q_944 = _f2_639;
                    float2 r_945 = fma_f32x2_rn_noftz(q_944, scale1_909, a_943);
                    acc_844[22] = r_945.x;
                    acc_844[23] = r_945.y;
                    float _unpack_fp4_47[8];
                    {
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
                        uint32_t _fp4_decode_pairs[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %4; \n"             " cvt.rn.bf16x2.e2m1x2 %0, __b0; \n"             " cvt.rn.bf16x2.e2m1x2 %1, __b1; \n"             " cvt.rn.bf16x2.e2m1x2 %2, __b2; \n"             " cvt.rn.bf16x2.e2m1x2 %3, __b3; \n"             " } \n"             : "=r"(_fp4_decode_pairs[0]), "=r"(_fp4_decode_pairs[1]), "=r"(_fp4_decode_pairs[2]), "=r"(_fp4_decode_pairs[3]) : "r"(r_853[3 + 0]));
                        (&_unpack_fp4_47[0])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) << 16);
                        (&_unpack_fp4_47[0])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[0]) & 0xffff0000u);
                        (&_unpack_fp4_47[2])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) << 16);
                        (&_unpack_fp4_47[2])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[1]) & 0xffff0000u);
                        (&_unpack_fp4_47[4])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) << 16);
                        (&_unpack_fp4_47[4])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[2]) & 0xffff0000u);
                        (&_unpack_fp4_47[6])[0] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) << 16);
                        (&_unpack_fp4_47[6])[1] = __uint_as_float(static_cast<uint32_t>(_fp4_decode_pairs[3]) & 0xffff0000u);
                        #else
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " .reg .b32 __hpair; \n"             " .reg .b16 __h0, __h1; \n"             " mov.b32 {__b0, __b1, __b2, __b3}, %8; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b0; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %0, __h0; \n"             " cvt.f32.f16 %1, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b1; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %2, __h0; \n"             " cvt.f32.f16 %3, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b2; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %4, __h0; \n"             " cvt.f32.f16 %5, __h1; \n"             " cvt.rn.f16x2.e2m1x2 __hpair, __b3; \n"             " mov.b32 {__h0, __h1}, __hpair; \n"             " cvt.f32.f16 %6, __h0; \n"             " cvt.f32.f16 %7, __h1; \n"             " } \n"             : "=f"(_unpack_fp4_47[0]), "=f"(_unpack_fp4_47[1]), "=f"(_unpack_fp4_47[2]), "=f"(_unpack_fp4_47[3]), "=f"(_unpack_fp4_47[4]), "=f"(_unpack_fp4_47[5]), "=f"(_unpack_fp4_47[6]), "=f"(_unpack_fp4_47[7]) : "r"(r_853[3 + 0]));
                        #endif
                    }
                    float2 _f2_640 = make_float2(acc_844[24], acc_844[25]);
                    float2 a_946 = _f2_640;
                    float2 _f2_641 = make_float2(_unpack_fp4_47[0], _unpack_fp4_47[1]);
                    float2 q_947 = _f2_641;
                    float2 r_948 = fma_f32x2_rn_noftz(q_947, scale1_909, a_946);
                    acc_844[24] = r_948.x;
                    acc_844[25] = r_948.y;
                    float2 _f2_642 = make_float2(acc_844[26], acc_844[27]);
                    float2 a_949 = _f2_642;
                    float2 _f2_643 = make_float2(_unpack_fp4_47[2], _unpack_fp4_47[3]);
                    float2 q_950 = _f2_643;
                    float2 r_951 = fma_f32x2_rn_noftz(q_950, scale1_909, a_949);
                    acc_844[26] = r_951.x;
                    acc_844[27] = r_951.y;
                    float2 _f2_644 = make_float2(acc_844[28], acc_844[29]);
                    float2 a_952 = _f2_644;
                    float2 _f2_645 = make_float2(_unpack_fp4_47[4], _unpack_fp4_47[5]);
                    float2 q_953 = _f2_645;
                    float2 r_954 = fma_f32x2_rn_noftz(q_953, scale1_909, a_952);
                    acc_844[28] = r_954.x;
                    acc_844[29] = r_954.y;
                    float2 _f2_646 = make_float2(acc_844[30], acc_844[31]);
                    float2 a_955 = _f2_646;
                    float2 _f2_647 = make_float2(_unpack_fp4_47[6], _unpack_fp4_47[7]);
                    float2 q_956 = _f2_647;
                    float2 r_957 = fma_f32x2_rn_noftz(q_956, scale1_909, a_955);
                    acc_844[30] = r_957.x;
                    acc_844[31] = r_957.y;
                    unsigned int t_958[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_958[0])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(0) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 0))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_958[4])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(4) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 1))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_958[8])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(8) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 2))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&t_958[12])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&t_958[(12) + 3]))
                        : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 3))));
                    float2 _f2_648 = make_float2(acc_844[0], acc_844[1]);
                    float2 a2_959 = _f2_648;
                    float2 _f2_649 = make_float2(__uint_as_float(t_958[0] << 16), __uint_as_float(t_958[0] & 4294901760u));
                    float2 z_960 = add_f32x2_noftz(a2_959, _f2_649);
                    acc_844[0] = z_960.x;
                    acc_844[1] = z_960.y;
                    float2 _f2_650 = make_float2(ssx, ssy);
                    float2 ss2_961 = fma_f32x2_rn_noftz(z_960, z_960, _f2_650);
                    ssx = ss2_961.x;
                    ssy = ss2_961.y;
                    float2 _f2_651 = make_float2(acc_844[2], acc_844[3]);
                    float2 a2_962 = _f2_651;
                    float2 _f2_652 = make_float2(__uint_as_float(t_958[1] << 16), __uint_as_float(t_958[1] & 4294901760u));
                    float2 z_963 = add_f32x2_noftz(a2_962, _f2_652);
                    acc_844[2] = z_963.x;
                    acc_844[3] = z_963.y;
                    float2 _f2_653 = make_float2(ssx, ssy);
                    float2 ss2_964 = fma_f32x2_rn_noftz(z_963, z_963, _f2_653);
                    ssx = ss2_964.x;
                    ssy = ss2_964.y;
                    float2 _f2_654 = make_float2(acc_844[4], acc_844[5]);
                    float2 a2_965 = _f2_654;
                    float2 _f2_655 = make_float2(__uint_as_float(t_958[2] << 16), __uint_as_float(t_958[2] & 4294901760u));
                    float2 z_966 = add_f32x2_noftz(a2_965, _f2_655);
                    acc_844[4] = z_966.x;
                    acc_844[5] = z_966.y;
                    float2 _f2_656 = make_float2(ssx, ssy);
                    float2 ss2_967 = fma_f32x2_rn_noftz(z_966, z_966, _f2_656);
                    ssx = ss2_967.x;
                    ssy = ss2_967.y;
                    float2 _f2_657 = make_float2(acc_844[6], acc_844[7]);
                    float2 a2_968 = _f2_657;
                    float2 _f2_658 = make_float2(__uint_as_float(t_958[3] << 16), __uint_as_float(t_958[3] & 4294901760u));
                    float2 z_969 = add_f32x2_noftz(a2_968, _f2_658);
                    acc_844[6] = z_969.x;
                    acc_844[7] = z_969.y;
                    float2 _f2_659 = make_float2(ssx, ssy);
                    float2 ss2_970 = fma_f32x2_rn_noftz(z_969, z_969, _f2_659);
                    ssx = ss2_970.x;
                    ssy = ss2_970.y;
                    float2 _f2_660 = make_float2(acc_844[8], acc_844[9]);
                    float2 a2_971 = _f2_660;
                    float2 _f2_661 = make_float2(__uint_as_float(t_958[4] << 16), __uint_as_float(t_958[4] & 4294901760u));
                    float2 z_972 = add_f32x2_noftz(a2_971, _f2_661);
                    acc_844[8] = z_972.x;
                    acc_844[9] = z_972.y;
                    float2 _f2_662 = make_float2(ssx, ssy);
                    float2 ss2_973 = fma_f32x2_rn_noftz(z_972, z_972, _f2_662);
                    ssx = ss2_973.x;
                    ssy = ss2_973.y;
                    float2 _f2_663 = make_float2(acc_844[10], acc_844[11]);
                    float2 a2_974 = _f2_663;
                    float2 _f2_664 = make_float2(__uint_as_float(t_958[5] << 16), __uint_as_float(t_958[5] & 4294901760u));
                    float2 z_975 = add_f32x2_noftz(a2_974, _f2_664);
                    acc_844[10] = z_975.x;
                    acc_844[11] = z_975.y;
                    float2 _f2_665 = make_float2(ssx, ssy);
                    float2 ss2_976 = fma_f32x2_rn_noftz(z_975, z_975, _f2_665);
                    ssx = ss2_976.x;
                    ssy = ss2_976.y;
                    float2 _f2_666 = make_float2(acc_844[12], acc_844[13]);
                    float2 a2_977 = _f2_666;
                    float2 _f2_667 = make_float2(__uint_as_float(t_958[6] << 16), __uint_as_float(t_958[6] & 4294901760u));
                    float2 z_978 = add_f32x2_noftz(a2_977, _f2_667);
                    acc_844[12] = z_978.x;
                    acc_844[13] = z_978.y;
                    float2 _f2_668 = make_float2(ssx, ssy);
                    float2 ss2_979 = fma_f32x2_rn_noftz(z_978, z_978, _f2_668);
                    ssx = ss2_979.x;
                    ssy = ss2_979.y;
                    float2 _f2_669 = make_float2(acc_844[14], acc_844[15]);
                    float2 a2_980 = _f2_669;
                    float2 _f2_670 = make_float2(__uint_as_float(t_958[7] << 16), __uint_as_float(t_958[7] & 4294901760u));
                    float2 z_981 = add_f32x2_noftz(a2_980, _f2_670);
                    acc_844[14] = z_981.x;
                    acc_844[15] = z_981.y;
                    float2 _f2_671 = make_float2(ssx, ssy);
                    float2 ss2_982 = fma_f32x2_rn_noftz(z_981, z_981, _f2_671);
                    ssx = ss2_982.x;
                    ssy = ss2_982.y;
                    float2 _f2_672 = make_float2(acc_844[16], acc_844[17]);
                    float2 a2_983 = _f2_672;
                    float2 _f2_673 = make_float2(__uint_as_float(t_958[8] << 16), __uint_as_float(t_958[8] & 4294901760u));
                    float2 z_984 = add_f32x2_noftz(a2_983, _f2_673);
                    acc_844[16] = z_984.x;
                    acc_844[17] = z_984.y;
                    float2 _f2_674 = make_float2(ssx, ssy);
                    float2 ss2_985 = fma_f32x2_rn_noftz(z_984, z_984, _f2_674);
                    ssx = ss2_985.x;
                    ssy = ss2_985.y;
                    float2 _f2_675 = make_float2(acc_844[18], acc_844[19]);
                    float2 a2_986 = _f2_675;
                    float2 _f2_676 = make_float2(__uint_as_float(t_958[9] << 16), __uint_as_float(t_958[9] & 4294901760u));
                    float2 z_987 = add_f32x2_noftz(a2_986, _f2_676);
                    acc_844[18] = z_987.x;
                    acc_844[19] = z_987.y;
                    float2 _f2_677 = make_float2(ssx, ssy);
                    float2 ss2_988 = fma_f32x2_rn_noftz(z_987, z_987, _f2_677);
                    ssx = ss2_988.x;
                    ssy = ss2_988.y;
                    float2 _f2_678 = make_float2(acc_844[20], acc_844[21]);
                    float2 a2_989 = _f2_678;
                    float2 _f2_679 = make_float2(__uint_as_float(t_958[10] << 16), __uint_as_float(t_958[10] & 4294901760u));
                    float2 z_990 = add_f32x2_noftz(a2_989, _f2_679);
                    acc_844[20] = z_990.x;
                    acc_844[21] = z_990.y;
                    float2 _f2_680 = make_float2(ssx, ssy);
                    float2 ss2_991 = fma_f32x2_rn_noftz(z_990, z_990, _f2_680);
                    ssx = ss2_991.x;
                    ssy = ss2_991.y;
                    float2 _f2_681 = make_float2(acc_844[22], acc_844[23]);
                    float2 a2_992 = _f2_681;
                    float2 _f2_682 = make_float2(__uint_as_float(t_958[11] << 16), __uint_as_float(t_958[11] & 4294901760u));
                    float2 z_993 = add_f32x2_noftz(a2_992, _f2_682);
                    acc_844[22] = z_993.x;
                    acc_844[23] = z_993.y;
                    float2 _f2_683 = make_float2(ssx, ssy);
                    float2 ss2_994 = fma_f32x2_rn_noftz(z_993, z_993, _f2_683);
                    ssx = ss2_994.x;
                    ssy = ss2_994.y;
                    float2 _f2_684 = make_float2(acc_844[24], acc_844[25]);
                    float2 a2_995 = _f2_684;
                    float2 _f2_685 = make_float2(__uint_as_float(t_958[12] << 16), __uint_as_float(t_958[12] & 4294901760u));
                    float2 z_996 = add_f32x2_noftz(a2_995, _f2_685);
                    acc_844[24] = z_996.x;
                    acc_844[25] = z_996.y;
                    float2 _f2_686 = make_float2(ssx, ssy);
                    float2 ss2_997 = fma_f32x2_rn_noftz(z_996, z_996, _f2_686);
                    ssx = ss2_997.x;
                    ssy = ss2_997.y;
                    float2 _f2_687 = make_float2(acc_844[26], acc_844[27]);
                    float2 a2_998 = _f2_687;
                    float2 _f2_688 = make_float2(__uint_as_float(t_958[13] << 16), __uint_as_float(t_958[13] & 4294901760u));
                    float2 z_999 = add_f32x2_noftz(a2_998, _f2_688);
                    acc_844[26] = z_999.x;
                    acc_844[27] = z_999.y;
                    float2 _f2_689 = make_float2(ssx, ssy);
                    float2 ss2_1000 = fma_f32x2_rn_noftz(z_999, z_999, _f2_689);
                    ssx = ss2_1000.x;
                    ssy = ss2_1000.y;
                    float2 _f2_690 = make_float2(acc_844[28], acc_844[29]);
                    float2 a2_1001 = _f2_690;
                    float2 _f2_691 = make_float2(__uint_as_float(t_958[14] << 16), __uint_as_float(t_958[14] & 4294901760u));
                    float2 z_1002 = add_f32x2_noftz(a2_1001, _f2_691);
                    acc_844[28] = z_1002.x;
                    acc_844[29] = z_1002.y;
                    float2 _f2_692 = make_float2(ssx, ssy);
                    float2 ss2_1003 = fma_f32x2_rn_noftz(z_1002, z_1002, _f2_692);
                    ssx = ss2_1003.x;
                    ssy = ss2_1003.y;
                    float2 _f2_693 = make_float2(acc_844[30], acc_844[31]);
                    float2 a2_1004 = _f2_693;
                    float2 _f2_694 = make_float2(__uint_as_float(t_958[15] << 16), __uint_as_float(t_958[15] & 4294901760u));
                    float2 z_1005 = add_f32x2_noftz(a2_1004, _f2_694);
                    acc_844[30] = z_1005.x;
                    acc_844[31] = z_1005.y;
                    float2 _f2_695 = make_float2(ssx, ssy);
                    float2 ss2_1006 = fma_f32x2_rn_noftz(z_1005, z_1005, _f2_695);
                    ssx = ss2_1006.x;
                    ssy = ss2_1006.y;
                    unsigned int acc_844_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_844[_lp*2 + 0], acc_844[_lp*2+1 + 0]));
                        acc_844_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 0))), "r"(acc_844_bf16[0]), "r"(acc_844_bf16[1]), "r"(acc_844_bf16[2]), "r"(acc_844_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 1))), "r"(acc_844_bf16[4]), "r"(acc_844_bf16[5]), "r"(acc_844_bf16[6]), "r"(acc_844_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 2))), "r"(acc_844_bf16[8]), "r"(acc_844_bf16[9]), "r"(acc_844_bf16[10]), "r"(acc_844_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_837 + 16 * (f ^ 3))), "r"(acc_844_bf16[12]), "r"(acc_844_bf16[13]), "r"(acc_844_bf16[14]), "r"(acc_844_bf16[15]) : "memory");
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
                    float _add_0;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(ssx), "f"(ssy));
                    float ss = _add_0;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ss, 16);
                    float _add_1;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(ss), "f"(_shfl_xor_0));
                    ss = _add_1;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ss, 8);
                    float _add_2;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(ss), "f"(_shfl_xor_1));
                    ss = _add_2;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ss, 4);
                    float _add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(ss), "f"(_shfl_xor_2));
                    ss = _add_3;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ss, 2);
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(ss), "f"(_shfl_xor_3));
                    ss = _add_4;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, ss, 1);
                    float _add_5;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(ss), "f"(_shfl_xor_4));
                    ss = _add_5;
                    float _fdiv_rn_0 = __fdiv_rn(ss, 6144.0f);
                    float _add_6;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(_fdiv_rn_0), "f"(eps));
                    float arg = _add_6;
                    float _rsqrt_0;
                    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(_rsqrt_0) : "f"(arg));
                    float rinv = _rsqrt_0;
                    float2 _f2_696 = make_float2(rinv, rinv);
                    float2 r2 = _f2_696;
                    float2 _f2_697 = make_float2(wbias, wbias);
                    float2 b2 = _f2_697;
                    float six = 6.0f;
                    float e4m3_max = 448.0f;
                    float _rcp_0 = approx_rcp(six);
                    float rcp6 = _rcp_0;
                    float _rcp_1 = approx_rcp(gso);
                    float rcp_gs = _rcp_1;
                    float _rcp_2 = approx_rcp(e4m3_max);
                    float rcp448 = _rcp_2;
                    #pragma unroll 1
                    for (int j = 0; j < 6; j++) {
                        int pp_6 = lane_2 + 32 * j;
                        unsigned int tx[16];
                        unsigned int tw[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[0])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(0) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_6 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_6 + 16 * (f ^ 0))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[4])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(4) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_6 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[4])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(4) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_6 + 16 * (f ^ 1))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[8])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(8) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_6 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[8])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(8) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_6 + 16 * (f ^ 2))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tx[12])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tx[(12) + 3]))
                            : "r"(wsm_addr + (unsigned int)(rb_off + 64 * pp_6 + 16 * (f ^ 3))));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw[12])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(12) + 3]))
                            : "r"(wrow_addr + (unsigned int)(64 * pp_6 + 16 * (f ^ 3))));
                        float y[32];
                        float2 _f2_698 = make_float2(__uint_as_float(tx[0] << 16), __uint_as_float(tx[0] & 4294901760u));
                        float2 tz;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz) : "l"(*(const unsigned long long*)&_f2_698), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_699 = make_float2(__uint_as_float(tw[0] << 16), __uint_as_float(tw[0] & 4294901760u));
                        float2 tw_7 = add_f32x2_noftz(b2, _f2_699);
                        float2 t_8;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&tz), "l"(*(const unsigned long long*)&tw_7));
                        y[0] = t_8.x;
                        y[1] = t_8.y;
                        float2 _f2_700 = make_float2(__uint_as_float(tx[1] << 16), __uint_as_float(tx[1] & 4294901760u));
                        float2 tz_9;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_9) : "l"(*(const unsigned long long*)&_f2_700), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_701 = make_float2(__uint_as_float(tw[1] << 16), __uint_as_float(tw[1] & 4294901760u));
                        float2 tw_10 = add_f32x2_noftz(b2, _f2_701);
                        float2 t_11;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&tz_9), "l"(*(const unsigned long long*)&tw_10));
                        y[2] = t_11.x;
                        y[3] = t_11.y;
                        float2 _f2_702 = make_float2(__uint_as_float(tx[2] << 16), __uint_as_float(tx[2] & 4294901760u));
                        float2 tz_12;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_12) : "l"(*(const unsigned long long*)&_f2_702), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_703 = make_float2(__uint_as_float(tw[2] << 16), __uint_as_float(tw[2] & 4294901760u));
                        float2 tw_13 = add_f32x2_noftz(b2, _f2_703);
                        float2 t_14;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&tz_12), "l"(*(const unsigned long long*)&tw_13));
                        y[4] = t_14.x;
                        y[5] = t_14.y;
                        float2 _f2_704 = make_float2(__uint_as_float(tx[3] << 16), __uint_as_float(tx[3] & 4294901760u));
                        float2 tz_15;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_15) : "l"(*(const unsigned long long*)&_f2_704), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_705 = make_float2(__uint_as_float(tw[3] << 16), __uint_as_float(tw[3] & 4294901760u));
                        float2 tw_16 = add_f32x2_noftz(b2, _f2_705);
                        float2 t_17;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&tz_15), "l"(*(const unsigned long long*)&tw_16));
                        y[6] = t_17.x;
                        y[7] = t_17.y;
                        float2 _f2_706 = make_float2(__uint_as_float(tx[4] << 16), __uint_as_float(tx[4] & 4294901760u));
                        float2 tz_18;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_18) : "l"(*(const unsigned long long*)&_f2_706), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_707 = make_float2(__uint_as_float(tw[4] << 16), __uint_as_float(tw[4] & 4294901760u));
                        float2 tw_19 = add_f32x2_noftz(b2, _f2_707);
                        float2 t_20;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&tz_18), "l"(*(const unsigned long long*)&tw_19));
                        y[8] = t_20.x;
                        y[9] = t_20.y;
                        float2 _f2_708 = make_float2(__uint_as_float(tx[5] << 16), __uint_as_float(tx[5] & 4294901760u));
                        float2 tz_21;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_21) : "l"(*(const unsigned long long*)&_f2_708), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_709 = make_float2(__uint_as_float(tw[5] << 16), __uint_as_float(tw[5] & 4294901760u));
                        float2 tw_22 = add_f32x2_noftz(b2, _f2_709);
                        float2 t_23;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&tz_21), "l"(*(const unsigned long long*)&tw_22));
                        y[10] = t_23.x;
                        y[11] = t_23.y;
                        float2 _f2_710 = make_float2(__uint_as_float(tx[6] << 16), __uint_as_float(tx[6] & 4294901760u));
                        float2 tz_24;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_24) : "l"(*(const unsigned long long*)&_f2_710), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_711 = make_float2(__uint_as_float(tw[6] << 16), __uint_as_float(tw[6] & 4294901760u));
                        float2 tw_25 = add_f32x2_noftz(b2, _f2_711);
                        float2 t_26;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&tz_24), "l"(*(const unsigned long long*)&tw_25));
                        y[12] = t_26.x;
                        y[13] = t_26.y;
                        float2 _f2_712 = make_float2(__uint_as_float(tx[7] << 16), __uint_as_float(tx[7] & 4294901760u));
                        float2 tz_27;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_27) : "l"(*(const unsigned long long*)&_f2_712), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_713 = make_float2(__uint_as_float(tw[7] << 16), __uint_as_float(tw[7] & 4294901760u));
                        float2 tw_28 = add_f32x2_noftz(b2, _f2_713);
                        float2 t_29;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&tz_27), "l"(*(const unsigned long long*)&tw_28));
                        y[14] = t_29.x;
                        y[15] = t_29.y;
                        float2 _f2_714 = make_float2(__uint_as_float(tx[8] << 16), __uint_as_float(tx[8] & 4294901760u));
                        float2 tz_30;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_30) : "l"(*(const unsigned long long*)&_f2_714), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_715 = make_float2(__uint_as_float(tw[8] << 16), __uint_as_float(tw[8] & 4294901760u));
                        float2 tw_31 = add_f32x2_noftz(b2, _f2_715);
                        float2 t_32;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_32) : "l"(*(const unsigned long long*)&tz_30), "l"(*(const unsigned long long*)&tw_31));
                        y[16] = t_32.x;
                        y[17] = t_32.y;
                        float2 _f2_716 = make_float2(__uint_as_float(tx[9] << 16), __uint_as_float(tx[9] & 4294901760u));
                        float2 tz_33;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_33) : "l"(*(const unsigned long long*)&_f2_716), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_717 = make_float2(__uint_as_float(tw[9] << 16), __uint_as_float(tw[9] & 4294901760u));
                        float2 tw_34 = add_f32x2_noftz(b2, _f2_717);
                        float2 t_35;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_35) : "l"(*(const unsigned long long*)&tz_33), "l"(*(const unsigned long long*)&tw_34));
                        y[18] = t_35.x;
                        y[19] = t_35.y;
                        float2 _f2_718 = make_float2(__uint_as_float(tx[10] << 16), __uint_as_float(tx[10] & 4294901760u));
                        float2 tz_36;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_36) : "l"(*(const unsigned long long*)&_f2_718), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_719 = make_float2(__uint_as_float(tw[10] << 16), __uint_as_float(tw[10] & 4294901760u));
                        float2 tw_37 = add_f32x2_noftz(b2, _f2_719);
                        float2 t_38;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_38) : "l"(*(const unsigned long long*)&tz_36), "l"(*(const unsigned long long*)&tw_37));
                        y[20] = t_38.x;
                        y[21] = t_38.y;
                        float2 _f2_720 = make_float2(__uint_as_float(tx[11] << 16), __uint_as_float(tx[11] & 4294901760u));
                        float2 tz_39;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_39) : "l"(*(const unsigned long long*)&_f2_720), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_721 = make_float2(__uint_as_float(tw[11] << 16), __uint_as_float(tw[11] & 4294901760u));
                        float2 tw_40 = add_f32x2_noftz(b2, _f2_721);
                        float2 t_41;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_41) : "l"(*(const unsigned long long*)&tz_39), "l"(*(const unsigned long long*)&tw_40));
                        y[22] = t_41.x;
                        y[23] = t_41.y;
                        float2 _f2_722 = make_float2(__uint_as_float(tx[12] << 16), __uint_as_float(tx[12] & 4294901760u));
                        float2 tz_42;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_42) : "l"(*(const unsigned long long*)&_f2_722), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_723 = make_float2(__uint_as_float(tw[12] << 16), __uint_as_float(tw[12] & 4294901760u));
                        float2 tw_43 = add_f32x2_noftz(b2, _f2_723);
                        float2 t_44;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_44) : "l"(*(const unsigned long long*)&tz_42), "l"(*(const unsigned long long*)&tw_43));
                        y[24] = t_44.x;
                        y[25] = t_44.y;
                        float2 _f2_724 = make_float2(__uint_as_float(tx[13] << 16), __uint_as_float(tx[13] & 4294901760u));
                        float2 tz_45;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_45) : "l"(*(const unsigned long long*)&_f2_724), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_725 = make_float2(__uint_as_float(tw[13] << 16), __uint_as_float(tw[13] & 4294901760u));
                        float2 tw_46 = add_f32x2_noftz(b2, _f2_725);
                        float2 t_47;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_47) : "l"(*(const unsigned long long*)&tz_45), "l"(*(const unsigned long long*)&tw_46));
                        y[26] = t_47.x;
                        y[27] = t_47.y;
                        float2 _f2_726 = make_float2(__uint_as_float(tx[14] << 16), __uint_as_float(tx[14] & 4294901760u));
                        float2 tz_48;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_48) : "l"(*(const unsigned long long*)&_f2_726), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_727 = make_float2(__uint_as_float(tw[14] << 16), __uint_as_float(tw[14] & 4294901760u));
                        float2 tw_49 = add_f32x2_noftz(b2, _f2_727);
                        float2 t_50;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_50) : "l"(*(const unsigned long long*)&tz_48), "l"(*(const unsigned long long*)&tw_49));
                        y[28] = t_50.x;
                        y[29] = t_50.y;
                        float2 _f2_728 = make_float2(__uint_as_float(tx[15] << 16), __uint_as_float(tx[15] & 4294901760u));
                        float2 tz_51;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&tz_51) : "l"(*(const unsigned long long*)&_f2_728), "l"(*(const unsigned long long*)&r2));
                        float2 _f2_729 = make_float2(__uint_as_float(tw[15] << 16), __uint_as_float(tw[15] & 4294901760u));
                        float2 tw_52 = add_f32x2_noftz(b2, _f2_729);
                        float2 t_53;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_53) : "l"(*(const unsigned long long*)&tz_51), "l"(*(const unsigned long long*)&tw_52));
                        y[30] = t_53.x;
                        y[31] = t_53.y;
                        unsigned int y_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                            y_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
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
                        float zero = 0.0f;
                        bool fodd_54 = (f & 1) != 0;
                        bool fhi_55 = (f & 2) != 0;
                        float _fabs_0;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_0) : "f"(yb[0]));
                        float amag0 = _fabs_0;
                        float _fabs_1;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_1) : "f"(yb[1]));
                        float amag1 = _fabs_1;
                        float _fabs_2;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_2) : "f"(yb[2]));
                        float amag2 = _fabs_2;
                        float _fabs_3;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_3) : "f"(yb[3]));
                        float amag3 = _fabs_3;
                        float _fabs_4;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_4) : "f"(yb[4]));
                        float amag4 = _fabs_4;
                        float _fabs_5;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_5) : "f"(yb[5]));
                        float amag5 = _fabs_5;
                        float _fabs_6;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_6) : "f"(yb[6]));
                        float amag6 = _fabs_6;
                        float _fabs_7;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_7) : "f"(yb[7]));
                        float amag7 = _fabs_7;
                        float _fabs_8;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_8) : "f"(yb[8]));
                        float amag8 = _fabs_8;
                        float _fabs_9;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_9) : "f"(yb[9]));
                        float amag9 = _fabs_9;
                        float _fabs_10;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_10) : "f"(yb[10]));
                        float amag10 = _fabs_10;
                        float _fabs_11;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_11) : "f"(yb[11]));
                        float amag11 = _fabs_11;
                        float _fabs_12;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_12) : "f"(yb[12]));
                        float amag12 = _fabs_12;
                        float _fabs_13;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_13) : "f"(yb[13]));
                        float amag13 = _fabs_13;
                        float _fabs_14;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_14) : "f"(yb[14]));
                        float amag14 = _fabs_14;
                        float _fabs_15;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_15) : "f"(yb[15]));
                        float amag15 = _fabs_15;
                        float _max3_0;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_0) : "f"(zero), "f"(amag0), "f"(amag1));
                        float amax0_0 = _max3_0;
                        float _max3_1;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(amag2), "f"(amag3), "f"(amag4));
                        float amax0_1 = _max3_1;
                        float _max3_2;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_2) : "f"(amag5), "f"(amag6), "f"(amag7));
                        float amax0_2 = _max3_2;
                        float _max3_3;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_3) : "f"(amag8), "f"(amag9), "f"(amag10));
                        float amax0_3 = _max3_3;
                        float _max3_4;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(amag11), "f"(amag12), "f"(amag13));
                        float amax0_4 = _max3_4;
                        float _max3_5;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(amag14), "f"(amag15), "f"(zero));
                        float amax0_5 = _max3_5;
                        float _max3_6;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(amax0_0), "f"(amax0_1), "f"(amax0_2));
                        float amax1_0 = _max3_6;
                        float _max3_7;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(amax0_3), "f"(amax0_4), "f"(amax0_5));
                        float amax1_1 = _max3_7;
                        float _max3_8;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(amax1_0), "f"(amax1_1), "f"(zero));
                        float amax2_0 = _max3_8;
                        float _fabs_16;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_16) : "f"(yb[16]));
                        float amag0_56 = _fabs_16;
                        float _fabs_17;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_17) : "f"(yb[17]));
                        float amag1_57 = _fabs_17;
                        float _fabs_18;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_18) : "f"(yb[18]));
                        float amag2_58 = _fabs_18;
                        float _fabs_19;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_19) : "f"(yb[19]));
                        float amag3_59 = _fabs_19;
                        float _fabs_20;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_20) : "f"(yb[20]));
                        float amag4_60 = _fabs_20;
                        float _fabs_21;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_21) : "f"(yb[21]));
                        float amag5_61 = _fabs_21;
                        float _fabs_22;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_22) : "f"(yb[22]));
                        float amag6_62 = _fabs_22;
                        float _fabs_23;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_23) : "f"(yb[23]));
                        float amag7_63 = _fabs_23;
                        float _fabs_24;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_24) : "f"(yb[24]));
                        float amag8_64 = _fabs_24;
                        float _fabs_25;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_25) : "f"(yb[25]));
                        float amag9_65 = _fabs_25;
                        float _fabs_26;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_26) : "f"(yb[26]));
                        float amag10_66 = _fabs_26;
                        float _fabs_27;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_27) : "f"(yb[27]));
                        float amag11_67 = _fabs_27;
                        float _fabs_28;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_28) : "f"(yb[28]));
                        float amag12_68 = _fabs_28;
                        float _fabs_29;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_29) : "f"(yb[29]));
                        float amag13_69 = _fabs_29;
                        float _fabs_30;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_30) : "f"(yb[30]));
                        float amag14_70 = _fabs_30;
                        float _fabs_31;
                        asm("abs.f32 %0, %1;" : "=f"(_fabs_31) : "f"(yb[31]));
                        float amag15_71 = _fabs_31;
                        float _max3_9;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(zero), "f"(amag0_56), "f"(amag1_57));
                        float amax0_0_72 = _max3_9;
                        float _max3_10;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(amag2_58), "f"(amag3_59), "f"(amag4_60));
                        float amax0_1_73 = _max3_10;
                        float _max3_11;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(amag5_61), "f"(amag6_62), "f"(amag7_63));
                        float amax0_2_74 = _max3_11;
                        float _max3_12;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(amag8_64), "f"(amag9_65), "f"(amag10_66));
                        float amax0_3_75 = _max3_12;
                        float _max3_13;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(amag11_67), "f"(amag12_68), "f"(amag13_69));
                        float amax0_4_76 = _max3_13;
                        float _max3_14;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(amag14_70), "f"(amag15_71), "f"(zero));
                        float amax0_5_77 = _max3_14;
                        float _max3_15;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_15) : "f"(amax0_0_72), "f"(amax0_1_73), "f"(amax0_2_74));
                        float amax1_0_78 = _max3_15;
                        float _max3_16;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(amax0_3_75), "f"(amax0_4_76), "f"(amax0_5_77));
                        float amax1_1_79 = _max3_16;
                        float _max3_17;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(amax1_0_78), "f"(amax1_1_79), "f"(zero));
                        float amax2_0_80 = _max3_17;
                        float vm0 = amax2_0;
                        float vm1 = amax2_0_80;
                        if (fhi_55) {
                            vm0 = amax2_0_80;
                            vm1 = amax2_0;
                        }
                        unsigned int sfb[2];
                        float _mul_24;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(vm0), "f"(rcp6));
                        float _mul_25;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(gso), "f"(_mul_24));
                        float sfv = _mul_25;
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(zero), "f"(sfv));
                        sfb[0] = (unsigned int)_e4m3x2_f32_0 & 255;
                        uint32_t _e4m3x2_to_f16x2_12;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_12) : "h"((uint16_t)(_e4m3x2_f32_0)));
                        float _cvt_f32_f16_24;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(_e4m3x2_to_f16x2_12)));
                        float _cvt_f32_f16_25;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(_e4m3x2_to_f16x2_12 >> 16)));
                        float _mul_26;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_cvt_f32_f16_24), "f"(rcp_gs));
                        float _rcp_3 = approx_rcp(_mul_26);
                        float oscale = 0.0f;
                        if (vm0 != 0.0f) {
                            oscale = _rcp_3;
                        }
                        float _mul_27;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(vm1), "f"(rcp6));
                        float _mul_28;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(gso), "f"(_mul_27));
                        float sfv_81 = _mul_28;
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(zero), "f"(sfv_81));
                        sfb[1] = (unsigned int)_e4m3x2_f32_1 & 255;
                        uint32_t _e4m3x2_to_f16x2_13;
                        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_e4m3x2_to_f16x2_13) : "h"((uint16_t)(_e4m3x2_f32_1)));
                        float _cvt_f32_f16_26;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(_e4m3x2_to_f16x2_13)));
                        float _cvt_f32_f16_27;
                        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(_e4m3x2_to_f16x2_13 >> 16)));
                        float _mul_29;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_cvt_f32_f16_26), "f"(rcp_gs));
                        float _rcp_4 = approx_rcp(_mul_29);
                        float oscale_82 = 0.0f;
                        if (vm1 != 0.0f) {
                            oscale_82 = _rcp_4;
                        }
                        float os_p0 = oscale;
                        float os_p1 = oscale_82;
                        if (fhi_55) {
                            os_p0 = oscale_82;
                            os_p1 = oscale;
                        }
                        unsigned int w_83[4];
                        float _mul_30;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(yb[0]), "f"(os_p0));
                        float _mul_31;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(yb[1]), "f"(os_p0));
                        uint32_t _fp4_pair_0;
                        uint16_t _e2m1_pair_h_9;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_9) : "f"(_mul_30), "f"(_mul_31));
                        _fp4_pair_0 = (uint32_t)_e2m1_pair_h_9;
                        float _mul_32;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(yb[2]), "f"(os_p0));
                        float _mul_33;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(yb[3]), "f"(os_p0));
                        uint32_t _fp4_pair_1;
                        uint16_t _e2m1_pair_h_10;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_10) : "f"(_mul_32), "f"(_mul_33));
                        _fp4_pair_1 = (uint32_t)_e2m1_pair_h_10;
                        float _mul_34;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(yb[4]), "f"(os_p0));
                        float _mul_35;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(yb[5]), "f"(os_p0));
                        uint32_t _fp4_pair_2;
                        uint16_t _e2m1_pair_h_11;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_11) : "f"(_mul_34), "f"(_mul_35));
                        _fp4_pair_2 = (uint32_t)_e2m1_pair_h_11;
                        float _mul_36;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(yb[6]), "f"(os_p0));
                        float _mul_37;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(yb[7]), "f"(os_p0));
                        uint32_t _fp4_pair_3;
                        uint16_t _e2m1_pair_h_12;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_12) : "f"(_mul_36), "f"(_mul_37));
                        _fp4_pair_3 = (uint32_t)_e2m1_pair_h_12;
                        w_83[0] = _fp4_pair_0 | _fp4_pair_1 << 8 | _fp4_pair_2 << 16 | _fp4_pair_3 << 24;
                        float _mul_38;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(yb[8]), "f"(os_p0));
                        float _mul_39;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(yb[9]), "f"(os_p0));
                        uint32_t _fp4_pair_4;
                        uint16_t _e2m1_pair_h_13;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_13) : "f"(_mul_38), "f"(_mul_39));
                        _fp4_pair_4 = (uint32_t)_e2m1_pair_h_13;
                        float _mul_40;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(yb[10]), "f"(os_p0));
                        float _mul_41;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(yb[11]), "f"(os_p0));
                        uint32_t _fp4_pair_5;
                        uint16_t _e2m1_pair_h_14;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_14) : "f"(_mul_40), "f"(_mul_41));
                        _fp4_pair_5 = (uint32_t)_e2m1_pair_h_14;
                        float _mul_42;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(yb[12]), "f"(os_p0));
                        float _mul_43;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(yb[13]), "f"(os_p0));
                        uint32_t _fp4_pair_6;
                        uint16_t _e2m1_pair_h_15;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_15) : "f"(_mul_42), "f"(_mul_43));
                        _fp4_pair_6 = (uint32_t)_e2m1_pair_h_15;
                        float _mul_44;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(yb[14]), "f"(os_p0));
                        float _mul_45;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(yb[15]), "f"(os_p0));
                        uint32_t _fp4_pair_7;
                        uint16_t _e2m1_pair_h_16;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_16) : "f"(_mul_44), "f"(_mul_45));
                        _fp4_pair_7 = (uint32_t)_e2m1_pair_h_16;
                        w_83[1] = _fp4_pair_4 | _fp4_pair_5 << 8 | _fp4_pair_6 << 16 | _fp4_pair_7 << 24;
                        float _mul_46;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(yb[16]), "f"(os_p1));
                        float _mul_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(yb[17]), "f"(os_p1));
                        uint32_t _fp4_pair_8;
                        uint16_t _e2m1_pair_h_17;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_17) : "f"(_mul_46), "f"(_mul_47));
                        _fp4_pair_8 = (uint32_t)_e2m1_pair_h_17;
                        float _mul_48;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(yb[18]), "f"(os_p1));
                        float _mul_49;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(yb[19]), "f"(os_p1));
                        uint32_t _fp4_pair_9;
                        uint16_t _e2m1_pair_h_18;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_18) : "f"(_mul_48), "f"(_mul_49));
                        _fp4_pair_9 = (uint32_t)_e2m1_pair_h_18;
                        float _mul_50;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(yb[20]), "f"(os_p1));
                        float _mul_51;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(yb[21]), "f"(os_p1));
                        uint32_t _fp4_pair_10;
                        uint16_t _e2m1_pair_h_19;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_19) : "f"(_mul_50), "f"(_mul_51));
                        _fp4_pair_10 = (uint32_t)_e2m1_pair_h_19;
                        float _mul_52;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(yb[22]), "f"(os_p1));
                        float _mul_53;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(yb[23]), "f"(os_p1));
                        uint32_t _fp4_pair_11;
                        uint16_t _e2m1_pair_h_20;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_20) : "f"(_mul_52), "f"(_mul_53));
                        _fp4_pair_11 = (uint32_t)_e2m1_pair_h_20;
                        w_83[2] = _fp4_pair_8 | _fp4_pair_9 << 8 | _fp4_pair_10 << 16 | _fp4_pair_11 << 24;
                        float _mul_54;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(yb[24]), "f"(os_p1));
                        float _mul_55;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(yb[25]), "f"(os_p1));
                        uint32_t _fp4_pair_12;
                        uint16_t _e2m1_pair_h_21;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_21) : "f"(_mul_54), "f"(_mul_55));
                        _fp4_pair_12 = (uint32_t)_e2m1_pair_h_21;
                        float _mul_56;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(yb[26]), "f"(os_p1));
                        float _mul_57;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(yb[27]), "f"(os_p1));
                        uint32_t _fp4_pair_13;
                        uint16_t _e2m1_pair_h_22;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_22) : "f"(_mul_56), "f"(_mul_57));
                        _fp4_pair_13 = (uint32_t)_e2m1_pair_h_22;
                        float _mul_58;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(yb[28]), "f"(os_p1));
                        float _mul_59;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(yb[29]), "f"(os_p1));
                        uint32_t _fp4_pair_14;
                        uint16_t _e2m1_pair_h_23;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_23) : "f"(_mul_58), "f"(_mul_59));
                        _fp4_pair_14 = (uint32_t)_e2m1_pair_h_23;
                        float _mul_60;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(yb[30]), "f"(os_p1));
                        float _mul_61;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(yb[31]), "f"(os_p1));
                        uint32_t _fp4_pair_15;
                        uint16_t _e2m1_pair_h_24;
                        asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b16 %0, {byte0, 0};\n"             "}\n"             : "=h"(_e2m1_pair_h_24) : "f"(_mul_60), "f"(_mul_61));
                        _fp4_pair_15 = (uint32_t)_e2m1_pair_h_24;
                        w_83[3] = _fp4_pair_12 | _fp4_pair_13 << 8 | _fp4_pair_14 << 16 | _fp4_pair_15 << 24;
                        unsigned int sw[4];
                        sw[0] = w_83[0];
                        sw[1] = w_83[1];
                        sw[2] = w_83[2];
                        sw[3] = w_83[3];
                        if (fodd_54) {
                            sw[0] = w_83[1];
                            sw[1] = w_83[0];
                            sw[2] = w_83[3];
                            sw[3] = w_83[2];
                        }
                        unsigned int words[4];
                        words[0] = sw[0];
                        words[1] = sw[1];
                        words[2] = sw[2];
                        words[3] = sw[3];
                        if (fhi_55) {
                            words[0] = sw[2];
                            words[1] = sw[3];
                            words[2] = sw[0];
                            words[3] = sw[1];
                        }
                        long long vidx = drow * 3072 + (long long)(16 * pp_6) >> 2;
                        {
                            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
                            *reinterpret_cast<int4*>(reinterpret_cast<int*>(outq) + vidx) = _iv4;
                        }
                        unsigned int half = sfb[0] | sfb[1] << 8;
                        unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, half, 1);
                        unsigned int oddmask = -(unsigned int)(lane_2 & 1);
                        unsigned int lo_84 = half & ~oddmask | _shfl_xor_5 & oddmask;
                        unsigned int hi_85 = _shfl_xor_5 & ~oddmask | half & oddmask;
                        unsigned int word = lo_84 | hi_85 << 16;
                        long long rowpart = drow / 128 * 49152 + drow % 32 * 16 + drow % 128 / 32 * 4;
                        int bpart = (4 * (pp_6 >> 1) >> 2) * 512 + (4 * (pp_6 >> 1) & 3);
                        long long off = rowpart + (long long)bpart;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(out_sf + off)) + (0)) = word;
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
            asm volatile("barrier.sync 6, 288;" ::: "memory");
            if (tid == 0) {
                __threadfence();
                unsigned int _atomic_old_0 = atomicAdd(&ctrl[1], 1);
                unsigned int last = (unsigned int)gridDim.x - 1;
                if (_atomic_old_0 == last) {
                    ctrl[1] = 0;
                    ctrl[2] = (unsigned int)rows;
                    ctrl[3] = 1;
                    __threadfence();
                    ctrl[0] = call + 1;
                    __threadfence();
                }
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
                long long lo_1 = u * bid64_1 / g64_1;
                long long hi_1 = u * (bid64_1 + 1) / g64_1;
                long long g0 = lo_1 + lane64;
                #pragma unroll 1
                for (long long g = g0; g < hi_1; g += 32) {
                    long long widx = v_base_1 + 16 * g >> 2;
                    {
                        int4 _iv4 = make_int4(sv4[0 + 0], sv4[0 + 1], sv4[0 + 2], sv4[0 + 3]);
                        *reinterpret_cast<int4*>(reinterpret_cast<int*>(os_wire) + widx) = _iv4;
                    }
                }
                long long u_0 = pr64 * 384 / 16;
                long long lo_1_1 = u_0 * bid64_1 / g64_1;
                long long hi_2 = u_0 * (bid64_1 + 1) / g64_1;
                long long g0_3 = lo_1_1 + lane64;
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
            asm volatile("barrier.sync 6, 288;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
