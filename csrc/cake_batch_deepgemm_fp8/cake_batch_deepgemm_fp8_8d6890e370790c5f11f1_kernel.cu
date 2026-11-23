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
#define TMEM_NCOLS 32
#define TMEM_ACCUM_OFFSET 0
#define NUM_ONE_SHOT_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_EPI_STAGING_OFF 21504
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_TOTAL 29696
#define THREADS 192

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};




__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ void tcgen05_commit_leader(
    int mbar_addr, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %1, 0;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(leader) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(192) void
kernel_cake_batch_deepgemm_fp8_8d6890e370790c5f11f1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B_tensor, const __grid_constant__ CUtensorMap C_tma, float* __restrict__ A_scale, float* __restrict__ B_scale, int* __restrict__ masked_m, unsigned int num_groups, unsigned int M, unsigned int num_n_tiles)
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
    #define tmem_full_addr (mbar_base + 8)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 21504);
    const int epi_staging_addr = smem + 21504;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_tensor))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");
    unsigned int k_tile_idx = (unsigned int)blockIdx.x;
    unsigned int k_lane = (unsigned int)lane;
    unsigned int k_has_tile = 0;
    unsigned int k_batch = 0;
    unsigned int k_start = 0;
    unsigned int k_base = 0;
    unsigned int k_zero = 0;
    unsigned int k_one = 1;
    unsigned int k_glast = num_groups - 1;
    #pragma unroll 1
    for (unsigned int k_chunk = 0; k_chunk < num_groups; k_chunk += 32) {
        unsigned int k_g = k_chunk + k_lane;
        unsigned int k_in = ((k_g < num_groups) ? k_one : k_zero);
        unsigned int k_gl = ((k_g < num_groups) ? k_g : k_glast);
        unsigned int k_m = (unsigned int)masked_m[k_gl];
        unsigned int k_t = k_in * ((k_m + 128 - 1) / 128) * num_n_tiles;
        uint32_t k_incl = k_t;
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, k_incl, 1, 32); if (lane >= 1) { k_incl += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, k_incl, 2, 32); if (lane >= 2) { k_incl += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, k_incl, 4, 32); if (lane >= 4) { k_incl += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, k_incl, 8, 32); if (lane >= 8) { k_incl += peer; } }
        { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, k_incl, 16, 32); if (lane >= 16) { k_incl += peer; } }
        unsigned int k_lo = k_base + k_incl - k_t;
        unsigned int k_hi = k_base + k_incl;
        unsigned int k_hit = 0;
        if (k_in != 0) {
            if (k_tile_idx >= k_lo) {
                if (k_tile_idx < k_hi) {
                    k_hit = 1;
                }
            }
        }
        unsigned int _warp_redux_u32_0;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(k_hit * (k_g + 1)));
        unsigned int _warp_redux_u32_1;
        asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(k_hit * (k_lo + 1)));
        if (k_has_tile == 0) {
            if (_warp_redux_u32_0 != 0) {
                k_batch = _warp_redux_u32_0 - 1;
                k_start = _warp_redux_u32_1 - 1;
                k_has_tile = 1;
            }
        }
        unsigned int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, k_incl, 31);
        k_base = k_base + _shfl_sync_0;
    }
    unsigned int k_safe = ((k_has_tile != 0) ? k_tile_idx : k_zero);
    unsigned int k_in_batch = k_safe - k_start;
    unsigned int k_m_tile = k_in_batch / num_n_tiles;
    unsigned int k_n_tile = k_in_batch - k_m_tile * num_n_tiles;
    unsigned int k_off_m = k_m_tile * 128;
    unsigned int k_off_n = k_n_tile * 32;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one_shot' ---
            // tma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // tmem_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel pre-TMEM ops
    if (k_has_tile == 0) {
        return;
    }

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 16);
    if (warp == 0) {
        int _tmem_hold = smem + 16;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            const int epi_warp = warp;
            int row = epi_warp * 32 + lane;
            float sa = A_scale[k_batch * M + k_off_m + (unsigned int)row];
            float sb = B_scale[k_batch * (num_n_tiles / 4) + k_n_tile / 4];
            float scale = sa * sb;
            unsigned int _phase_tmem_full_0 = 0;
            mbarrier_wait(tmem_full_addr, _phase_tmem_full_0);
            _phase_tmem_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (int chunk = 0; chunk < 4; chunk++) {
                int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(chunk * 8);
                float _tmem_load_0[8];
                tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float values_scaled[8];
                const float2 _scale2_0 = {scale, scale};
                #pragma unroll
                for (int _ls = 0; _ls < 4; _ls++)
                    reinterpret_cast<float2*>(values_scaled)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                unsigned int values_scaled_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_scaled[_lp*2 + 0], values_scaled[_lp*2+1 + 0]));
                    values_scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                __nv_bfloat16* _sv_ptr = reinterpret_cast<__nv_bfloat16*>(epi_staging + (row * 32 + chunk * 8));
                reinterpret_cast<int4*>(_sv_ptr + 0)[0] = reinterpret_cast<int4*>(values_scaled_bf16)[0];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (warp == 0) {
                if (elect_sync()) {
                    if (k_has_tile != 0) {
                        tma_store_3d((&C_tma), k_off_n, k_off_m, k_batch, epi_staging_addr);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("barrier.sync 15, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(32));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            if (elect_sync()) {
                tma_4d_gmem2smem(smem_a_addr, (&A), 0, k_off_m, 0, k_batch, tma_full_addr);
                tma_4d_gmem2smem(smem_b_addr, (&B_tensor), 0, k_off_n, 0, k_batch, tma_full_addr);
                mbarrier_arrive_expect_tx(tma_full_addr, 20480);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int _phase_tma_full_0 = 0;
            mbarrier_wait(tma_full_addr, _phase_tma_full_0);
            _phase_tma_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t _elected_0 = elect_sync();
            int _smem_desc_lo_1 = ((smem_a_addr) >> 4) & 0x3FFF;
            int _smem_desc_lo_2 = ((smem_b_addr) >> 4) & 0x3FFF;
            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_1), "r"(_smem_desc_lo_2), "r"(tmem_accum), "r"(0), "r"(_elected_0) : "memory");
            tcgen05_commit_leader(tmem_full_addr, _elected_0);
        }
    }

    // Cleanup
}

} // extern "C"
