/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_dense_projection_det_gemm_9d269d61e56ad3d54fae(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ out, int M, int N, int K, int B, int ldo, int out_l)
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
    #define tma_full_addr (mbar_base + 0)
    #define tma_empty_addr (mbar_base + 96)
    #define mma_epi_full_addr (mbar_base + 192)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_a_addr = smem + 99328;
    unsigned int* staging = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int staging_addr = smem + 197632;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (warp == 1) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 25 barriers)
    // Mbarriers at smem_raw[0..200)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 12 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // tma_empty: 12 barriers, init_count=1
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
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // --- pipeline 'epi_pipe' ---
            // mma_epi_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 200);
    if (warp == 2) {
        int _tmem_hold = smem + 200;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(160) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: dma_w ----
    if (warp == 0) {
        { // dma_w_main
            unsigned int w_stage = 0;
            int w_idx = blockIdx.x * 64;
            int b_idx = blockIdx.z;
            int k_tiles = K / 64;
            int k_begin = 0;
            unsigned int _phase_tma_empty = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 4
                for (int k_tile = k_begin; k_tile < k_begin + k_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (w_stage) * 8, _phase_tma_empty);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(smem_w_addr + w_stage * 8192), "l"((&W)), "r"(0), "r"(w_idx), "r"(k_tile), "r"(b_idx),
                           "r"(tma_full_addr + (w_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(tma_full_addr + (w_stage) * 8, 8192);
                    w_stage += 1;
                    if (w_stage == 12) { w_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: dma_a ----
    if (warp == 1) {
        { // dma_a_main
            unsigned int a_stage = 0;
            int t_idx = blockIdx.y * 64;
            int b_idx_a = blockIdx.z;
            int k_tiles_a = K / 64;
            int k_begin_a = 0;
            unsigned int _phase_tma_empty_1 = 1;
            if (elect_sync()) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 4
                for (int k_tile_1 = k_begin_a; k_tile_1 < k_begin_a + k_tiles_a; k_tile_1++) {
                    mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty_1);
                    tma_4d_gmem2smem(smem_a_addr + a_stage * 8192, (&A), 0, t_idx, k_tile_1, b_idx_a, tma_full_addr + (a_stage) * 8);
                    mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 8192);
                    a_stage += 1;
                    if (a_stage == 12) { a_stage = 0; _phase_tma_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 2) {
        { // mma_main
            unsigned int mma_stage = 0;
            int k_tiles_m = K / 64;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 2
            for (int k_tile_2 = 0; k_tile_2 < k_tiles_m; k_tile_2++) {
                mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int init_flag = ((k_tile_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                int _mma_a_lo_0 = make_warp_uniform((((smem_w_addr) >> 4) & 0x7FFF) + (mma_stage) * 512);
                int _mma_b_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 512);
#else
                int _mma_a_lo_0 = make_warp_uniform((((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 512);
                int _mma_b_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 512);
#endif
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, ((init_flag) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_accum, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 68158608, 1);
                    }
                }
                elect_commit(tma_empty_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 12) { mma_stage = 0; _phase_tma_full ^= 1; }
            }
            elect_commit(mma_epi_full_addr);
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            int row_local = epi_warp * 16 + lane % 16;
            int col_half = lane / 16;
            int n_row = blockIdx.x * 64 + row_local;
            int m_first = blockIdx.y * 64 + col_half * 32;
            int b_idx_e = blockIdx.z;
            mbarrier_wait(mma_epi_full_addr, 0);
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float _tmem_load_0[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                : "r"(tmem_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            {
                __nv_bfloat16 _bval_0 = __float2bfloat16_rn(_tmem_load_0[0]);
                uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                uint32_t _addr_0 = static_cast<uint32_t>(staging_addr + (unsigned int)((col_half * 32 * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0));
            }
            {
                __nv_bfloat16 _bval_1 = __float2bfloat16_rn(_tmem_load_0[1]);
                uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                uint32_t _addr_1 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 1) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1));
            }
            {
                __nv_bfloat16 _bval_2 = __float2bfloat16_rn(_tmem_load_0[2]);
                uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                uint32_t _addr_2 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 2) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2));
            }
            {
                __nv_bfloat16 _bval_3 = __float2bfloat16_rn(_tmem_load_0[3]);
                uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                uint32_t _addr_3 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 3) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3));
            }
            {
                __nv_bfloat16 _bval_4 = __float2bfloat16_rn(_tmem_load_0[4]);
                uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                uint32_t _addr_4 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 4) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4));
            }
            {
                __nv_bfloat16 _bval_5 = __float2bfloat16_rn(_tmem_load_0[5]);
                uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                uint32_t _addr_5 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 5) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5));
            }
            {
                __nv_bfloat16 _bval_6 = __float2bfloat16_rn(_tmem_load_0[6]);
                uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                uint32_t _addr_6 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 6) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6));
            }
            {
                __nv_bfloat16 _bval_7 = __float2bfloat16_rn(_tmem_load_0[7]);
                uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                uint32_t _addr_7 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 7) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7));
            }
            {
                __nv_bfloat16 _bval_8 = __float2bfloat16_rn(_tmem_load_0[8]);
                uint16_t _bits_8 = *(uint16_t*)&_bval_8;
                uint32_t _addr_8 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 8) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8));
            }
            {
                __nv_bfloat16 _bval_9 = __float2bfloat16_rn(_tmem_load_0[9]);
                uint16_t _bits_9 = *(uint16_t*)&_bval_9;
                uint32_t _addr_9 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 9) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_9), "h"(_bits_9));
            }
            {
                __nv_bfloat16 _bval_10 = __float2bfloat16_rn(_tmem_load_0[10]);
                uint16_t _bits_10 = *(uint16_t*)&_bval_10;
                uint32_t _addr_10 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 10) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_10), "h"(_bits_10));
            }
            {
                __nv_bfloat16 _bval_11 = __float2bfloat16_rn(_tmem_load_0[11]);
                uint16_t _bits_11 = *(uint16_t*)&_bval_11;
                uint32_t _addr_11 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 11) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_11), "h"(_bits_11));
            }
            {
                __nv_bfloat16 _bval_12 = __float2bfloat16_rn(_tmem_load_0[12]);
                uint16_t _bits_12 = *(uint16_t*)&_bval_12;
                uint32_t _addr_12 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 12) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_12), "h"(_bits_12));
            }
            {
                __nv_bfloat16 _bval_13 = __float2bfloat16_rn(_tmem_load_0[13]);
                uint16_t _bits_13 = *(uint16_t*)&_bval_13;
                uint32_t _addr_13 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 13) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_13), "h"(_bits_13));
            }
            {
                __nv_bfloat16 _bval_14 = __float2bfloat16_rn(_tmem_load_0[14]);
                uint16_t _bits_14 = *(uint16_t*)&_bval_14;
                uint32_t _addr_14 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 14) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14));
            }
            {
                __nv_bfloat16 _bval_15 = __float2bfloat16_rn(_tmem_load_0[15]);
                uint16_t _bits_15 = *(uint16_t*)&_bval_15;
                uint32_t _addr_15 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 15) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_15), "h"(_bits_15));
            }
            {
                __nv_bfloat16 _bval_16 = __float2bfloat16_rn(_tmem_load_0[16]);
                uint16_t _bits_16 = *(uint16_t*)&_bval_16;
                uint32_t _addr_16 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 16) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_16), "h"(_bits_16));
            }
            {
                __nv_bfloat16 _bval_17 = __float2bfloat16_rn(_tmem_load_0[17]);
                uint16_t _bits_17 = *(uint16_t*)&_bval_17;
                uint32_t _addr_17 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 17) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_17), "h"(_bits_17));
            }
            {
                __nv_bfloat16 _bval_18 = __float2bfloat16_rn(_tmem_load_0[18]);
                uint16_t _bits_18 = *(uint16_t*)&_bval_18;
                uint32_t _addr_18 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 18) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_18), "h"(_bits_18));
            }
            {
                __nv_bfloat16 _bval_19 = __float2bfloat16_rn(_tmem_load_0[19]);
                uint16_t _bits_19 = *(uint16_t*)&_bval_19;
                uint32_t _addr_19 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 19) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_19), "h"(_bits_19));
            }
            {
                __nv_bfloat16 _bval_20 = __float2bfloat16_rn(_tmem_load_0[20]);
                uint16_t _bits_20 = *(uint16_t*)&_bval_20;
                uint32_t _addr_20 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 20) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_20), "h"(_bits_20));
            }
            {
                __nv_bfloat16 _bval_21 = __float2bfloat16_rn(_tmem_load_0[21]);
                uint16_t _bits_21 = *(uint16_t*)&_bval_21;
                uint32_t _addr_21 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 21) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_21), "h"(_bits_21));
            }
            {
                __nv_bfloat16 _bval_22 = __float2bfloat16_rn(_tmem_load_0[22]);
                uint16_t _bits_22 = *(uint16_t*)&_bval_22;
                uint32_t _addr_22 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 22) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_22), "h"(_bits_22));
            }
            {
                __nv_bfloat16 _bval_23 = __float2bfloat16_rn(_tmem_load_0[23]);
                uint16_t _bits_23 = *(uint16_t*)&_bval_23;
                uint32_t _addr_23 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 23) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_23), "h"(_bits_23));
            }
            {
                __nv_bfloat16 _bval_24 = __float2bfloat16_rn(_tmem_load_0[24]);
                uint16_t _bits_24 = *(uint16_t*)&_bval_24;
                uint32_t _addr_24 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 24) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_24), "h"(_bits_24));
            }
            {
                __nv_bfloat16 _bval_25 = __float2bfloat16_rn(_tmem_load_0[25]);
                uint16_t _bits_25 = *(uint16_t*)&_bval_25;
                uint32_t _addr_25 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 25) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_25), "h"(_bits_25));
            }
            {
                __nv_bfloat16 _bval_26 = __float2bfloat16_rn(_tmem_load_0[26]);
                uint16_t _bits_26 = *(uint16_t*)&_bval_26;
                uint32_t _addr_26 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 26) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_26), "h"(_bits_26));
            }
            {
                __nv_bfloat16 _bval_27 = __float2bfloat16_rn(_tmem_load_0[27]);
                uint16_t _bits_27 = *(uint16_t*)&_bval_27;
                uint32_t _addr_27 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 27) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_27), "h"(_bits_27));
            }
            {
                __nv_bfloat16 _bval_28 = __float2bfloat16_rn(_tmem_load_0[28]);
                uint16_t _bits_28 = *(uint16_t*)&_bval_28;
                uint32_t _addr_28 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 28) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_28), "h"(_bits_28));
            }
            {
                __nv_bfloat16 _bval_29 = __float2bfloat16_rn(_tmem_load_0[29]);
                uint16_t _bits_29 = *(uint16_t*)&_bval_29;
                uint32_t _addr_29 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 29) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_29), "h"(_bits_29));
            }
            {
                __nv_bfloat16 _bval_30 = __float2bfloat16_rn(_tmem_load_0[30]);
                uint16_t _bits_30 = *(uint16_t*)&_bval_30;
                uint32_t _addr_30 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 30) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_30), "h"(_bits_30));
            }
            {
                __nv_bfloat16 _bval_31 = __float2bfloat16_rn(_tmem_load_0[31]);
                uint16_t _bits_31 = *(uint16_t*)&_bval_31;
                uint32_t _addr_31 = static_cast<uint32_t>(staging_addr + (unsigned int)(((col_half * 32 + 31) * 64 + row_local) * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_31), "h"(_bits_31));
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("bar.sync 2, 128;" ::: "memory");
            int c = epi_tid;
            int t_row = c / 8;
            int m_row = blockIdx.y * 64 + t_row;
            int n_col = blockIdx.x * 64 + c % 8 * 8;
            if (m_row < M && n_col < N) {
                unsigned int words[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(c * 16)));
                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + (((long long)b_idx_e * (long long)out_l + (long long)m_row * (long long)ldo + (long long)n_col) * 2)))[0] = reinterpret_cast<int4*>(words)[0];
            }
            int c_0 = epi_tid + 128;
            int t_row_0 = c_0 / 8;
            int m_row_0 = blockIdx.y * 64 + t_row_0;
            int n_col_0 = blockIdx.x * 64 + c_0 % 8 * 8;
            if (m_row_0 < M && n_col_0 < N) {
                unsigned int words_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(c_0 * 16)));
                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + (((long long)b_idx_e * (long long)out_l + (long long)m_row_0 * (long long)ldo + (long long)n_col_0) * 2)))[0] = reinterpret_cast<int4*>(words_1)[0];
            }
            int c_1 = epi_tid + 256;
            int t_row_1 = c_1 / 8;
            int m_row_1 = blockIdx.y * 64 + t_row_1;
            int n_col_1 = blockIdx.x * 64 + c_1 % 8 * 8;
            if (m_row_1 < M && n_col_1 < N) {
                unsigned int words_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(c_1 * 16)));
                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + (((long long)b_idx_e * (long long)out_l + (long long)m_row_1 * (long long)ldo + (long long)n_col_1) * 2)))[0] = reinterpret_cast<int4*>(words_2)[0];
            }
            int c_2 = epi_tid + 384;
            int t_row_2 = c_2 / 8;
            int m_row_2 = blockIdx.y * 64 + t_row_2;
            int n_col_2 = blockIdx.x * 64 + c_2 % 8 * 8;
            if (m_row_2 < M && n_col_2 < N) {
                unsigned int words_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
                    : "r"(staging_addr + (unsigned int)(c_2 * 16)));
                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(out + (((long long)b_idx_e * (long long)out_l + (long long)m_row_2 * (long long)ldo + (long long)n_col_2) * 2)))[0] = reinterpret_cast<int4*>(words_3)[0];
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"
