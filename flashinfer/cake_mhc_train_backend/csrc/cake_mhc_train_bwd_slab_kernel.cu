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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCW_OFFSET 0
#define TMEM_DFB0_OFFSET 256
#define TMEM_DFB1_OFFSET 320
#define TMEM_DFS0_OFFSET 384
#define TMEM_DFS1_OFFSET 448
#define NUM_STAGE_PIPE_STAGES 2
#define NUM_FN_PIPE_STAGES 1
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_X0_OFF 1024
#define SMEM_X0_STAGE_BYTES 8192
#define SMEM_X0_STRIDE 86016
#define SMEM_DY0_OFF 33792
#define SMEM_DY0_STAGE_BYTES 8192
#define SMEM_DY0_STRIDE 86016
#define SMEM_X1_OFF 9216
#define SMEM_X1_STAGE_BYTES 8192
#define SMEM_X1_STRIDE 86016
#define SMEM_DY1_OFF 41984
#define SMEM_DY1_STAGE_BYTES 8192
#define SMEM_DY1_STRIDE 86016
#define SMEM_X2_OFF 17408
#define SMEM_X2_STAGE_BYTES 8192
#define SMEM_X2_STRIDE 86016
#define SMEM_DY2_OFF 50176
#define SMEM_DY2_STAGE_BYTES 8192
#define SMEM_DY2_STRIDE 86016
#define SMEM_X3_OFF 25600
#define SMEM_X3_STAGE_BYTES 8192
#define SMEM_X3_STRIDE 86016
#define SMEM_DY3_OFF 58368
#define SMEM_DY3_STAGE_BYTES 8192
#define SMEM_DY3_STRIDE 86016
#define SMEM_DUS_OFF 66560
#define SMEM_DUS_STAGE_BYTES 8192
#define SMEM_DUS_STRIDE 86016
#define SMEM_DZ0_OFF 74752
#define SMEM_DZ0_STAGE_BYTES 4096
#define SMEM_DZ0_STRIDE 86016
#define SMEM_DZ1_OFF 78848
#define SMEM_DZ1_STAGE_BYTES 4096
#define SMEM_DZ1_STRIDE 86016
#define SMEM_DZ2_OFF 82944
#define SMEM_DZ2_STAGE_BYTES 4096
#define SMEM_DZ2_STRIDE 86016
#define SMEM_FN0_0_OFF 173056
#define SMEM_FN0_0_STAGE_BYTES 8192
#define SMEM_FN0_0_STRIDE 8192
#define SMEM_FN0_1_OFF 181248
#define SMEM_FN0_1_STAGE_BYTES 8192
#define SMEM_FN0_1_STRIDE 8192
#define SMEM_FN1_0_OFF 189440
#define SMEM_FN1_0_STAGE_BYTES 8192
#define SMEM_FN1_0_STRIDE 8192
#define SMEM_FN1_1_OFF 197632
#define SMEM_FN1_1_STAGE_BYTES 8192
#define SMEM_FN1_1_STRIDE 8192
#define SMEM_FN2_0_OFF 205824
#define SMEM_FN2_0_STAGE_BYTES 8192
#define SMEM_FN2_0_STRIDE 8192
#define SMEM_FN2_1_OFF 214016
#define SMEM_FN2_1_STAGE_BYTES 8192
#define SMEM_FN2_1_STRIDE 8192
#define SMEM_FNW0_OFF 173056
#define SMEM_FNW0_STAGE_BYTES 16384
#define SMEM_FNW0_STRIDE 16384
#define SMEM_FNW1_OFF 189440
#define SMEM_FNW1_STAGE_BYTES 16384
#define SMEM_FNW1_STRIDE 16384
#define SMEM_FNW2_OFF 205824
#define SMEM_FNW2_STAGE_BYTES 16384
#define SMEM_FNW2_STRIDE 16384
#define SMEM_XA0_OFF 1024
#define SMEM_XA0_STAGE_BYTES 16384
#define SMEM_XA0_STRIDE 86016
#define SMEM_XA1_OFF 17408
#define SMEM_XA1_STAGE_BYTES 16384
#define SMEM_XA1_STRIDE 86016
#define SMEM_DZN0_OFF 74752
#define SMEM_DZN0_STAGE_BYTES 4096
#define SMEM_DZN0_STRIDE 86016
#define SMEM_DZN1_OFF 78848
#define SMEM_DZN1_STAGE_BYTES 4096
#define SMEM_DZN1_STRIDE 86016
#define SMEM_DZN2_OFF 82944
#define SMEM_DZN2_STAGE_BYTES 4096
#define SMEM_DZN2_STRIDE 86016
#define SMEM_TOTAL 222208
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}






__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_mhc_train_bwd_slab(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap dY, const __grid_constant__ CUtensorMap dU, const __grid_constant__ CUtensorMap dzp, const __grid_constant__ CUtensorMap fnTp, const __grid_constant__ CUtensorMap dX, float* __restrict__ tokp, float* __restrict__ dfn_part, int T, int P, int rpc)
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
    #define stage_full_addr (mbar_base + 0)
    #define stage_empty_addr (mbar_base + 16)
    #define fn_full_addr (mbar_base + 32)
    #define fn_empty_addr (mbar_base + 40)
    #define acc_full_addr (mbar_base + 48)
    #define acc_empty_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* x0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int x0_addr = smem + 1024;
    __nv_bfloat16* dy0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dy0_addr = smem + 33792;
    __nv_bfloat16* x1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int x1_addr = smem + 9216;
    __nv_bfloat16* dy1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int dy1_addr = smem + 41984;
    __nv_bfloat16* x2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int x2_addr = smem + 17408;
    __nv_bfloat16* dy2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int dy2_addr = smem + 50176;
    __nv_bfloat16* x3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int x3_addr = smem + 25600;
    __nv_bfloat16* dy3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int dy3_addr = smem + 58368;
    __nv_bfloat16* dus = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dus_addr = smem + 66560;
    __nv_bfloat16* dz0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int dz0_addr = smem + 74752;
    __nv_bfloat16* dz1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 78848);
    const int dz1_addr = smem + 78848;
    __nv_bfloat16* dz2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int dz2_addr = smem + 82944;
    __nv_bfloat16* fn0_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int fn0_0_addr = smem + 173056;
    __nv_bfloat16* fn0_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int fn0_1_addr = smem + 181248;
    __nv_bfloat16* fn1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int fn1_0_addr = smem + 189440;
    __nv_bfloat16* fn1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int fn1_1_addr = smem + 197632;
    __nv_bfloat16* fn2_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int fn2_0_addr = smem + 205824;
    __nv_bfloat16* fn2_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int fn2_1_addr = smem + 214016;
    __nv_bfloat16* fnw0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int fnw0_addr = smem + 173056;
    __nv_bfloat16* fnw1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int fnw1_addr = smem + 189440;
    __nv_bfloat16* fnw2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int fnw2_addr = smem + 205824;
    __nv_bfloat16* xa0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int xa0_addr = smem + 1024;
    __nv_bfloat16* xa1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int xa1_addr = smem + 17408;
    __nv_bfloat16* dzn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int dzn0_addr = smem + 74752;
    __nv_bfloat16* dzn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 78848);
    const int dzn1_addr = smem + 78848;
    __nv_bfloat16* dzn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int dzn2_addr = smem + 82944;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dY))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dU))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dzp))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&fnTp))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dX))) : "memory");

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'stage_pipe' ---
            // stage_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // stage_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'fn_pipe' ---
            // fn_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // fn_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // acc_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accw = taddr;
    const int tmem_dfb0 = taddr + 256;
    const int tmem_dfb1 = taddr + 320;
    const int tmem_dfs0 = taddr + 384;
    const int tmem_dfs1 = taddr + 448;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
    }

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int _phase_fn_empty = 1;
            unsigned int _phase_stage_empty = 1;
            if (elect_sync()) {
                const int bid_0 = (int)bid;
                const int grid = (int)num_bids;
                const int n_mine = (64 * P - bid_0 + grid - 1) / grid;
                unsigned int ld_stage = 0;
                unsigned int fn_ld = 0;
                int first_ld = 1;
                int tcl = 0;
                #pragma unroll 1
                for (int j = 0; j < n_mine; j++) {
                    int u = bid_0 + j * grid;
                    int s = u / P;
                    int p = u - s * P;
                    int row0 = p * rpc;
                    int rows = T - row0;
                    if (rows > rpc) {
                        rows = rpc;
                    }
                    int n_tiles = 0;
                    if (rows > 0) {
                        n_tiles = (rows + 63) / 64;
                    }
                    if (n_tiles > 0) {
                        int k0 = s * 64;
                        if (j == 0) {
                            if (n_tiles > 0) {
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 4096)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 4096)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 8192)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 8192)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 12288)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 12288)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dU))), "r"((int)(k0)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(0)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(32)), "r"((int)(row0)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(64)), "r"((int)(row0)) : "memory");
                            }
                        }
                        mbarrier_wait(fn_empty_addr + (fn_ld) * 8, _phase_fn_empty);
                        tma_3d_gmem2smem(fn0_0_addr, (&fnTp), 0, k0, 0, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn0_0_addr + 4096, (&fnTp), 0, k0 + 4096, 0, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn0_1_addr, (&fnTp), 0, k0 + 8192, 0, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn0_1_addr + 4096, (&fnTp), 0, k0 + 12288, 0, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn1_0_addr, (&fnTp), 0, k0, 1, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn1_0_addr + 4096, (&fnTp), 0, k0 + 4096, 1, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn1_1_addr, (&fnTp), 0, k0 + 8192, 1, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn1_1_addr + 4096, (&fnTp), 0, k0 + 12288, 1, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn2_0_addr, (&fnTp), 0, k0, 2, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn2_0_addr + 4096, (&fnTp), 0, k0 + 4096, 2, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn2_1_addr, (&fnTp), 0, k0 + 8192, 2, fn_full_addr + (fn_ld) * 8);
                        tma_3d_gmem2smem(fn2_1_addr + 4096, (&fnTp), 0, k0 + 12288, 2, fn_full_addr + (fn_ld) * 8);
                        mbarrier_arrive_expect_tx(fn_full_addr + (fn_ld) * 8, 49152);
                        _phase_fn_empty ^= 1;
                        #pragma unroll 1
                        for (int tile = 0; tile < n_tiles; tile++) {
                            mbarrier_wait(stage_empty_addr + (ld_stage) * 8, _phase_stage_empty);
                            int t0 = row0 + tile * 64;
                            tma_2d_gmem2smem(x0_addr + ld_stage * 86016, (&X), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr + ld_stage * 86016, (&dY), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x1_addr + ld_stage * 86016, (&X), k0 + 4096, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy1_addr + ld_stage * 86016, (&dY), k0 + 4096, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x2_addr + ld_stage * 86016, (&X), k0 + 8192, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy2_addr + ld_stage * 86016, (&dY), k0 + 8192, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x3_addr + ld_stage * 86016, (&X), k0 + 12288, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy3_addr + ld_stage * 86016, (&dY), k0 + 12288, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dus_addr + ld_stage * 86016, (&dU), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz0_addr + ld_stage * 86016, (&dzp), 0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz1_addr + ld_stage * 86016, (&dzp), 32, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz2_addr + ld_stage * 86016, (&dzp), 64, t0, stage_full_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_addr + (ld_stage) * 8, 86016);
                            int nt = tile + 1;
                            if (nt < n_tiles) {
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 4096)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 4096)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 8192)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 8192)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(k0 + 12288)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(k0 + 12288)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dU))), "r"((int)(k0)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(0)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(32)), "r"((int)(row0 + nt * 64)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(64)), "r"((int)(row0 + nt * 64)) : "memory");
                            } else {
                                int j1 = j + 1;
                                if (j1 < n_mine) {
                                    int u_0 = bid_0 + j1 * grid;
                                    int s_1 = u_0 / P;
                                    int p_2 = u_0 - s_1 * P;
                                    int row0_3 = p_2 * rpc;
                                    int rows_4 = T - row0_3;
                                    if (rows_4 > rpc) {
                                        rows_4 = rpc;
                                    }
                                    int n_tiles_5 = 0;
                                    if (rows_4 > 0) {
                                        n_tiles_5 = (rows_4 + 63) / 64;
                                    }
                                    int tt = nt - n_tiles;
                                    if (tt < n_tiles_5) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_1 * 64)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_1 * 64)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_1 * 64 + 4096)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_1 * 64 + 4096)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_1 * 64 + 8192)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_1 * 64 + 8192)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_1 * 64 + 12288)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_1 * 64 + 12288)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dU))), "r"((int)(s_1 * 64)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(0)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(32)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(64)), "r"((int)(row0_3 + tt * 64)) : "memory");
                                    }
                                }
                            }
                            ld_stage += 1;
                            if (ld_stage == 2) { ld_stage = 0; _phase_stage_empty ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int bid_0_1 = (int)bid;
            const int grid_1 = (int)num_bids;
            const int n_mine_1 = (64 * P - bid_0_1 + grid_1 - 1) / grid_1;
            unsigned int mm_stage = 0;
            unsigned int acc_mm = 0;
            unsigned int fn_mm = 0;
            unsigned int grp_mm = 0;
            int tcm = 0;
            unsigned int _phase_fn_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_stage_full = 0;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < n_mine_1; j_1++) {
                int u_1 = bid_0_1 + j_1 * grid_1;
                int s_2 = u_1 / P;
                int p_1 = u_1 - s_2 * P;
                int row0_1 = p_1 * rpc;
                int rows_1 = T - row0_1;
                if (rows_1 > rpc) {
                    rows_1 = rpc;
                }
                int n_tiles_1 = 0;
                if (rows_1 > 0) {
                    n_tiles_1 = (rows_1 + 63) / 64;
                }
                if (n_tiles_1 > 0) {
                    mbarrier_wait(fn_full_addr + (fn_mm) * 8, _phase_fn_full);
                    #pragma unroll 1
                    for (int tile_1 = 0; tile_1 < n_tiles_1; tile_1++) {
                        mbarrier_wait(acc_empty_addr + (acc_mm) * 8, _phase_acc_empty);
                        mbarrier_wait(stage_full_addr + (mm_stage) * 8, _phase_stage_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_0 = make_warp_uniform((((dz0_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_0 = make_warp_uniform(((fnw2_addr) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_0 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_0 = make_warp_uniform(((fnw2_addr) >> 4) & 0x3FFF);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accw + (acc_mm * 128))), "r"(0));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = make_warp_uniform((((dz2_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_1 = make_warp_uniform(((fnw0_addr) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_1 = make_warp_uniform((((dz2_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_1 = make_warp_uniform(((fnw0_addr) >> 4) & 0x3FFF);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_2 = make_warp_uniform((((dz1_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_2 = make_warp_uniform(((fnw1_addr) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_2 = make_warp_uniform((((dz1_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_2 = make_warp_uniform(((fnw1_addr) >> 4) & 0x3FFF);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_accw + (acc_mm * 128))), "r"(1));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_3 = make_warp_uniform((((dz0_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_3 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_2), "r"((tmem_accw + (acc_mm * 128))), "r"(1));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_4 = make_warp_uniform((((dz1_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_4 = make_warp_uniform((((dz1_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_5 = make_warp_uniform((((dz0_addr) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_5 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1));
                        int in_grp = tile_1 % 16;
                        int small_init = ((in_grp == 0) ? 1 : 0);
                        unsigned int sbuf = grp_mm % 2;
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_6 = make_warp_uniform(((((xa0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_6 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_6 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_6 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_dfb0 + (acc_mm * 32))), "r"(0));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_7 = make_warp_uniform(((((xa0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_7 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_7 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_7 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_dfs0 + (sbuf * 32))), "r"(((small_init) ? 0 : 1)));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_8 = make_warp_uniform(((((xa0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_8 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_8 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_8 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"((tmem_dfs0 + (sbuf * 32))), "r"(1));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_9 = make_warp_uniform(((((xa1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_9 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_9 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_9 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_dfb1 + (acc_mm * 32))), "r"(0));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_10 = make_warp_uniform(((((xa1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_10 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_10 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_10 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_dfs1 + (sbuf * 32))), "r"(((small_init) ? 0 : 1)));
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_11 = make_warp_uniform(((((xa1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_11 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_11 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_11 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_dfs1 + (sbuf * 32))), "r"(1));
                        int bump = 0;
                        if (in_grp == 15) {
                            bump = 1;
                        }
                        if (tile_1 == n_tiles_1 - 1) {
                            bump = 1;
                        }
                        if (bump == 1) {
                            grp_mm = grp_mm + 1;
                        }
                        elect_commit(acc_full_addr + (acc_mm) * 8);
                        acc_mm += 1;
                        if (acc_mm == 2) { acc_mm = 0; _phase_acc_empty ^= 1; }
                        mm_stage += 1;
                        if (mm_stage == 2) { mm_stage = 0; _phase_stage_full ^= 1; }
                    }
                    elect_commit(fn_empty_addr + (fn_mm) * 8);
                    _phase_fn_full ^= 1;
                }
            }
        }
    }
    // ---- Role: spare ----
    if (warp >= 2 && warp <= 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epi ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // epi_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int bid_0_2 = (int)bid;
            const int grid_2 = (int)num_bids;
            const int n_mine_2 = (64 * P - bid_0_2 + grid_2 - 1) / grid_2;
            unsigned int ep_stage = 0;
            const int ew = warp - 4;
            int tce = 0;
            const int mt = ew / 4;
            const int q = warp % 4;
            const int lane_1 = lane;
            const int dl = q % 2 * 32 + lane_1;
            const int i_str = mt * 2 + q / 2;
            const int etid = ew * 32 + lane_1;
            unsigned int acc_ep = 0;
            unsigned int grp_ep = 0;
            bool _elect_sync_0 = elect_sync();
            int pend = 0;
            unsigned int pend_stage = 0;
            const int colw = dl / 2 * 4;
            const unsigned int sh = (unsigned int)(dl % 2 * 16);
            const unsigned int hb = (unsigned int)(dl % 2 * 2);
            int colsw[8];
            #pragma unroll
            for (int rr = 0; rr < 8; rr++) {
                colsw[rr] = colw ^ rr << 4;
            }
            unsigned int lane_tm = taddr + (unsigned int)(q * 32 << 16) + (unsigned int)(mt * 128);
            const unsigned int tmw = taddr + (unsigned int)(q * 32 + mt * 16 << 16);
            const int i0 = q / 2 * 2;
            const int tq = lane_1 / 4;
            const int lm4 = lane_1 % 4;
            const int tok0 = q % 2 * 32 + mt * 16 + tq;
            unsigned int offg[8];
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                offg[g] = (unsigned int)(((g ^ tq) << 4) + lm4 * 4);
            }
            const unsigned int rowb0 = (unsigned int)(tok0 * 128);
            const unsigned int trel0 = (unsigned int)(tok0 * 112);
            const unsigned int xoff0 = (unsigned int)(i0 * 8192);
            unsigned int lane_tb = taddr + (unsigned int)(q * 32 << 16) + (unsigned int)(256 + mt * 64);
            unsigned int lane_ts = taddr + (unsigned int)(q * 32 << 16) + (unsigned int)(384 + mt * 64);
            const unsigned long long pre_off = (unsigned long long)(2 + i_str);
            const unsigned long long c_off0 = (unsigned long long)(6 + 4 * i_str);
            const unsigned long long c_off1 = (unsigned long long)(8 + 4 * i_str);
            const int fr = etid / 3;
            const int fp = etid - fr * 3;
            const unsigned long long fq = (unsigned long long)(fp * 8);
            const unsigned int fdst = (unsigned int)(fr * 112 + (fr >> 3 & 1) * 16);
            unsigned int _phase_stage_full_1 = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int j_2 = 0; j_2 < n_mine_2; j_2++) {
                int u_2 = bid_0_2 + j_2 * grid_2;
                int s_3 = u_2 / P;
                int p_3 = u_2 - s_3 * P;
                int row0_2 = p_3 * rpc;
                int rows_2 = T - row0_2;
                if (rows_2 > rpc) {
                    rows_2 = rpc;
                }
                int n_tiles_2 = 0;
                if (rows_2 > 0) {
                    n_tiles_2 = (rows_2 + 63) / 64;
                }
                float acc[24];
                float comp[24];
                #pragma unroll
                for (int n = 0; n < 24; n++) {
                    acc[n] = 0.0f;
                    comp[n] = 0.0f;
                }
                float qa[8];
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < n_tiles_2; tile_2++) {
                    int t0_1 = row0_2 + tile_2 * 64;
                    mbarrier_wait(stage_full_addr + (ep_stage) * 8, _phase_stage_full_1);
                    if (tile_2 == 0) {
                        if (etid < 192) {
                            int rowa = t0_1 + fr;
                            if (rowa >= T) {
                                rowa = T - 1;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(tokp + (unsigned long long)rowa * 24 + fq);
                                qa[0 + 0] = _v4.x;
                                qa[0 + 1] = _v4.y;
                                qa[0 + 2] = _v4.z;
                                qa[0 + 3] = _v4.w;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(tokp + (unsigned long long)rowa * 24 + fq + 4);
                                qa[4 + 0] = _v4.x;
                                qa[4 + 1] = _v4.y;
                                qa[4 + 2] = _v4.z;
                                qa[4 + 3] = _v4.w;
                            }
                        }
                    }
                    mbarrier_wait(acc_full_addr + (acc_ep) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int tm = lane_tm + acc_ep * 64;
                    unsigned int tab = dz0_addr + ep_stage * 86016;
                    if (etid < 192) {
                        unsigned int fb = tab + fdst;
                        if (fp == 0) {
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fb + 64), "f"(qa[1]), "f"(qa[2]), "f"(qa[1]), "f"(qa[3]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fb + 80), "f"(qa[1]), "f"(qa[4]), "f"(qa[1]), "f"(qa[5]) : "memory");
                            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(fb), "f"(qa[6]), "f"(qa[7]) : "memory");
                        }
                        if (fp == 1) {
                            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(fb + 8), "f"(qa[0]), "f"(qa[1]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fb + 16), "f"(qa[2]), "f"(qa[3]), "f"(qa[4]), "f"(qa[5]) : "memory");
                            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(fb + 32), "f"(qa[6]), "f"(qa[7]) : "memory");
                        }
                        if (fp == 2) {
                            asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(fb + 40), "f"(qa[0]), "f"(qa[1]) : "memory");
                            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(fb + 48), "f"(qa[2]), "f"(qa[3]), "f"(qa[4]), "f"(qa[5]) : "memory");
                        }
                    }
                    unsigned int tmc = acc_ep * 128;
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(tmw + tmc));
                    float _tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                        : "r"(tmw + tmc + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (ew == 0) {
                        if (pend == 1) {
                            if (_elect_sync_0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                                mbarrier_arrive(stage_empty_addr + (pend_stage) * 8);
                            }
                            pend = 0;
                        }
                    }
                    unsigned int sbase = x0_addr + ep_stage * 86016;
                    unsigned int tv[24];
                    unsigned int trow = tab + trel0;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(0) + 1]))
                        : "r"(trow + (unsigned int)(i0 * 8 + 64)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(2) + 3]))
                        : "r"(trow + (unsigned int)(i0 * 16)));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(6) + 1]))
                        : "r"(trow + (unsigned int)((i0 + 1) * 8 + 64)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(8) + 3]))
                        : "r"(trow + (unsigned int)((i0 + 1) * 16)));
                    unsigned int trow_0 = tab + trel0 + 912;
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(12) + 1]))
                        : "r"(trow_0 + (unsigned int)(i0 * 8 + 64)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[14])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(14) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(14) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(14) + 3]))
                        : "r"(trow_0 + (unsigned int)(i0 * 16)));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[18])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(18) + 1]))
                        : "r"(trow_0 + (unsigned int)((i0 + 1) * 8 + 64)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[20])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(20) + 3]))
                        : "r"(trow_0 + (unsigned int)((i0 + 1) * 16)));
                    float nct[2];
                    nct[0] = -__uint_as_float(tv[0]);
                    nct[1] = -__uint_as_float(tv[12]);
                    unsigned int wv0[5];
                    unsigned int wv1[5];
                    unsigned int xr0[2];
                    unsigned int xr1[2];
                    unsigned int rb0 = sbase + offg[0] + rowb0;
                    unsigned int rb1 = rb0 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1 + 57344));
                    unsigned int xw0 = rb0 + xoff0;
                    unsigned int xw1 = rb1 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1 + 8192));
                    float dul = __uint_as_float(wv0[0] << 16);
                    float duh = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl[4];
                    float dyh[4];
                    dyl[0] = __uint_as_float(wv0[1] << 16);
                    dyh[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl[1] = __uint_as_float(wv0[2] << 16);
                    dyh[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl[2] = __uint_as_float(wv0[3] << 16);
                    dyh[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl[3] = __uint_as_float(wv0[4] << 16);
                    dyh[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i = __uint_as_float(tv[1]);
                    float c0 = __uint_as_float(tv[2]);
                    float c1 = __uint_as_float(tv[3]);
                    float c2 = __uint_as_float(tv[4]);
                    float c3 = __uint_as_float(tv[5]);
                    float _fma_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[0]));
                    float rl0 = _fma_0;
                    float _fma_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[1]));
                    float rh0 = _fma_1;
                    float _fma_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(pre_i), "f"(dul), "f"(rl0));
                    float rl1 = _fma_2;
                    float _fma_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(pre_i), "f"(duh), "f"(rh0));
                    float rh1 = _fma_3;
                    float _fma_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(c0), "f"(dyl[0]), "f"(rl1));
                    float rl2 = _fma_4;
                    float _fma_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(c0), "f"(dyh[0]), "f"(rh1));
                    float rh2 = _fma_5;
                    float _fma_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(c1), "f"(dyl[1]), "f"(rl2));
                    float rl3 = _fma_6;
                    float _fma_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(c1), "f"(dyh[1]), "f"(rh2));
                    float rh3 = _fma_7;
                    float _fma_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(c2), "f"(dyl[2]), "f"(rl3));
                    float rl4 = _fma_8;
                    float _fma_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(c2), "f"(dyh[2]), "f"(rh3));
                    float rh4 = _fma_9;
                    float _fma_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(c3), "f"(dyl[3]), "f"(rl4));
                    float rl5 = _fma_10;
                    float _fma_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(c3), "f"(dyh[3]), "f"(rh4));
                    float rh5 = _fma_11;
                    __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(rl5, rh5));
                    unsigned int word = __as_u32(_bf16x2_0);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0), "r"(word) : "memory");
                    float pre_i_1 = __uint_as_float(tv[7]);
                    float c0_2 = __uint_as_float(tv[8]);
                    float c1_3 = __uint_as_float(tv[9]);
                    float c2_4 = __uint_as_float(tv[10]);
                    float c3_5 = __uint_as_float(tv[11]);
                    float _fma_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[0]));
                    float rl0_6 = _fma_12;
                    float _fma_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[1]));
                    float rh0_7 = _fma_13;
                    float _fma_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(pre_i_1), "f"(dul), "f"(rl0_6));
                    float rl1_8 = _fma_14;
                    float _fma_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(pre_i_1), "f"(duh), "f"(rh0_7));
                    float rh1_9 = _fma_15;
                    float _fma_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(c0_2), "f"(dyl[0]), "f"(rl1_8));
                    float rl2_10 = _fma_16;
                    float _fma_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(c0_2), "f"(dyh[0]), "f"(rh1_9));
                    float rh2_11 = _fma_17;
                    float _fma_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(c1_3), "f"(dyl[1]), "f"(rl2_10));
                    float rl3_12 = _fma_18;
                    float _fma_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(c1_3), "f"(dyh[1]), "f"(rh2_11));
                    float rh3_13 = _fma_19;
                    float _fma_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(c2_4), "f"(dyl[2]), "f"(rl3_12));
                    float rl4_14 = _fma_20;
                    float _fma_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(c2_4), "f"(dyh[2]), "f"(rh3_13));
                    float rh4_15 = _fma_21;
                    float _fma_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(c3_5), "f"(dyl[3]), "f"(rl4_14));
                    float rl5_16 = _fma_22;
                    float _fma_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(c3_5), "f"(dyh[3]), "f"(rh4_15));
                    float rh5_17 = _fma_23;
                    __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(rl5_16, rh5_17));
                    unsigned int word_18 = __as_u32(_bf16x2_1);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0 + 8192), "r"(word_18) : "memory");
                    float dul_19 = __uint_as_float(wv1[0] << 16);
                    float duh_20 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_21[4];
                    float dyh_22[4];
                    dyl_21[0] = __uint_as_float(wv1[1] << 16);
                    dyh_22[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_21[1] = __uint_as_float(wv1[2] << 16);
                    dyh_22[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_21[2] = __uint_as_float(wv1[3] << 16);
                    dyh_22[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_21[3] = __uint_as_float(wv1[4] << 16);
                    dyh_22[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_23 = __uint_as_float(tv[13]);
                    float c0_24 = __uint_as_float(tv[14]);
                    float c1_25 = __uint_as_float(tv[15]);
                    float c2_26 = __uint_as_float(tv[16]);
                    float c3_27 = __uint_as_float(tv[17]);
                    float _fma_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[2]));
                    float rl0_28 = _fma_24;
                    float _fma_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[3]));
                    float rh0_29 = _fma_25;
                    float _fma_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(pre_i_23), "f"(dul_19), "f"(rl0_28));
                    float rl1_30 = _fma_26;
                    float _fma_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(pre_i_23), "f"(duh_20), "f"(rh0_29));
                    float rh1_31 = _fma_27;
                    float _fma_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(c0_24), "f"(dyl_21[0]), "f"(rl1_30));
                    float rl2_32 = _fma_28;
                    float _fma_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(c0_24), "f"(dyh_22[0]), "f"(rh1_31));
                    float rh2_33 = _fma_29;
                    float _fma_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(c1_25), "f"(dyl_21[1]), "f"(rl2_32));
                    float rl3_34 = _fma_30;
                    float _fma_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(c1_25), "f"(dyh_22[1]), "f"(rh2_33));
                    float rh3_35 = _fma_31;
                    float _fma_32;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(c2_26), "f"(dyl_21[2]), "f"(rl3_34));
                    float rl4_36 = _fma_32;
                    float _fma_33;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(c2_26), "f"(dyh_22[2]), "f"(rh3_35));
                    float rh4_37 = _fma_33;
                    float _fma_34;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(c3_27), "f"(dyl_21[3]), "f"(rl4_36));
                    float rl5_38 = _fma_34;
                    float _fma_35;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(c3_27), "f"(dyh_22[3]), "f"(rh4_37));
                    float rh5_39 = _fma_35;
                    __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(rl5_38, rh5_39));
                    unsigned int word_40 = __as_u32(_bf16x2_2);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1), "r"(word_40) : "memory");
                    float pre_i_41 = __uint_as_float(tv[19]);
                    float c0_42 = __uint_as_float(tv[20]);
                    float c1_43 = __uint_as_float(tv[21]);
                    float c2_44 = __uint_as_float(tv[22]);
                    float c3_45 = __uint_as_float(tv[23]);
                    float _fma_36;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[2]));
                    float rl0_46 = _fma_36;
                    float _fma_37;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[3]));
                    float rh0_47 = _fma_37;
                    float _fma_38;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(pre_i_41), "f"(dul_19), "f"(rl0_46));
                    float rl1_48 = _fma_38;
                    float _fma_39;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(pre_i_41), "f"(duh_20), "f"(rh0_47));
                    float rh1_49 = _fma_39;
                    float _fma_40;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(c0_42), "f"(dyl_21[0]), "f"(rl1_48));
                    float rl2_50 = _fma_40;
                    float _fma_41;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(c0_42), "f"(dyh_22[0]), "f"(rh1_49));
                    float rh2_51 = _fma_41;
                    float _fma_42;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(c1_43), "f"(dyl_21[1]), "f"(rl2_50));
                    float rl3_52 = _fma_42;
                    float _fma_43;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(c1_43), "f"(dyh_22[1]), "f"(rh2_51));
                    float rh3_53 = _fma_43;
                    float _fma_44;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(c2_44), "f"(dyl_21[2]), "f"(rl3_52));
                    float rl4_54 = _fma_44;
                    float _fma_45;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(c2_44), "f"(dyh_22[2]), "f"(rh3_53));
                    float rh4_55 = _fma_45;
                    float _fma_46;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(c3_45), "f"(dyl_21[3]), "f"(rl4_54));
                    float rl5_56 = _fma_46;
                    float _fma_47;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(c3_45), "f"(dyh_22[3]), "f"(rh4_55));
                    float rh5_57 = _fma_47;
                    __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(rl5_56, rh5_57));
                    unsigned int word_58 = __as_u32(_bf16x2_3);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1 + 8192), "r"(word_58) : "memory");
                    unsigned int rb0_59 = sbase + offg[1] + rowb0;
                    unsigned int rb1_60 = rb0_59 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_59 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_60 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_59 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_60 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_59 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_60 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_59 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_60 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_59 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_60 + 57344));
                    unsigned int xw0_61 = rb0_59 + xoff0;
                    unsigned int xw1_62 = rb1_60 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_61));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_62));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_61 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_62 + 8192));
                    float dul_63 = __uint_as_float(wv0[0] << 16);
                    float duh_64 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_65[4];
                    float dyh_66[4];
                    dyl_65[0] = __uint_as_float(wv0[1] << 16);
                    dyh_66[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_65[1] = __uint_as_float(wv0[2] << 16);
                    dyh_66[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_65[2] = __uint_as_float(wv0[3] << 16);
                    dyh_66[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_65[3] = __uint_as_float(wv0[4] << 16);
                    dyh_66[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_67 = __uint_as_float(tv[1]);
                    float c0_68 = __uint_as_float(tv[2]);
                    float c1_69 = __uint_as_float(tv[3]);
                    float c2_70 = __uint_as_float(tv[4]);
                    float c3_71 = __uint_as_float(tv[5]);
                    float _fma_48;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[4]));
                    float rl0_72 = _fma_48;
                    float _fma_49;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[5]));
                    float rh0_73 = _fma_49;
                    float _fma_50;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(pre_i_67), "f"(dul_63), "f"(rl0_72));
                    float rl1_74 = _fma_50;
                    float _fma_51;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(pre_i_67), "f"(duh_64), "f"(rh0_73));
                    float rh1_75 = _fma_51;
                    float _fma_52;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(c0_68), "f"(dyl_65[0]), "f"(rl1_74));
                    float rl2_76 = _fma_52;
                    float _fma_53;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(c0_68), "f"(dyh_66[0]), "f"(rh1_75));
                    float rh2_77 = _fma_53;
                    float _fma_54;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(c1_69), "f"(dyl_65[1]), "f"(rl2_76));
                    float rl3_78 = _fma_54;
                    float _fma_55;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(c1_69), "f"(dyh_66[1]), "f"(rh2_77));
                    float rh3_79 = _fma_55;
                    float _fma_56;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(c2_70), "f"(dyl_65[2]), "f"(rl3_78));
                    float rl4_80 = _fma_56;
                    float _fma_57;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(c2_70), "f"(dyh_66[2]), "f"(rh3_79));
                    float rh4_81 = _fma_57;
                    float _fma_58;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(c3_71), "f"(dyl_65[3]), "f"(rl4_80));
                    float rl5_82 = _fma_58;
                    float _fma_59;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(c3_71), "f"(dyh_66[3]), "f"(rh4_81));
                    float rh5_83 = _fma_59;
                    __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(rl5_82, rh5_83));
                    unsigned int word_84 = __as_u32(_bf16x2_4);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_61), "r"(word_84) : "memory");
                    float pre_i_85 = __uint_as_float(tv[7]);
                    float c0_86 = __uint_as_float(tv[8]);
                    float c1_87 = __uint_as_float(tv[9]);
                    float c2_88 = __uint_as_float(tv[10]);
                    float c3_89 = __uint_as_float(tv[11]);
                    float _fma_60;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[4]));
                    float rl0_90 = _fma_60;
                    float _fma_61;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[5]));
                    float rh0_91 = _fma_61;
                    float _fma_62;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(pre_i_85), "f"(dul_63), "f"(rl0_90));
                    float rl1_92 = _fma_62;
                    float _fma_63;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(pre_i_85), "f"(duh_64), "f"(rh0_91));
                    float rh1_93 = _fma_63;
                    float _fma_64;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(c0_86), "f"(dyl_65[0]), "f"(rl1_92));
                    float rl2_94 = _fma_64;
                    float _fma_65;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(c0_86), "f"(dyh_66[0]), "f"(rh1_93));
                    float rh2_95 = _fma_65;
                    float _fma_66;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(c1_87), "f"(dyl_65[1]), "f"(rl2_94));
                    float rl3_96 = _fma_66;
                    float _fma_67;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(c1_87), "f"(dyh_66[1]), "f"(rh2_95));
                    float rh3_97 = _fma_67;
                    float _fma_68;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(c2_88), "f"(dyl_65[2]), "f"(rl3_96));
                    float rl4_98 = _fma_68;
                    float _fma_69;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(c2_88), "f"(dyh_66[2]), "f"(rh3_97));
                    float rh4_99 = _fma_69;
                    float _fma_70;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(c3_89), "f"(dyl_65[3]), "f"(rl4_98));
                    float rl5_100 = _fma_70;
                    float _fma_71;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(c3_89), "f"(dyh_66[3]), "f"(rh4_99));
                    float rh5_101 = _fma_71;
                    __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(rl5_100, rh5_101));
                    unsigned int word_102 = __as_u32(_bf16x2_5);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_61 + 8192), "r"(word_102) : "memory");
                    float dul_103 = __uint_as_float(wv1[0] << 16);
                    float duh_104 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_105[4];
                    float dyh_106[4];
                    dyl_105[0] = __uint_as_float(wv1[1] << 16);
                    dyh_106[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_105[1] = __uint_as_float(wv1[2] << 16);
                    dyh_106[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_105[2] = __uint_as_float(wv1[3] << 16);
                    dyh_106[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_105[3] = __uint_as_float(wv1[4] << 16);
                    dyh_106[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_107 = __uint_as_float(tv[13]);
                    float c0_108 = __uint_as_float(tv[14]);
                    float c1_109 = __uint_as_float(tv[15]);
                    float c2_110 = __uint_as_float(tv[16]);
                    float c3_111 = __uint_as_float(tv[17]);
                    float _fma_72;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[6]));
                    float rl0_112 = _fma_72;
                    float _fma_73;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[7]));
                    float rh0_113 = _fma_73;
                    float _fma_74;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(pre_i_107), "f"(dul_103), "f"(rl0_112));
                    float rl1_114 = _fma_74;
                    float _fma_75;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(pre_i_107), "f"(duh_104), "f"(rh0_113));
                    float rh1_115 = _fma_75;
                    float _fma_76;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(c0_108), "f"(dyl_105[0]), "f"(rl1_114));
                    float rl2_116 = _fma_76;
                    float _fma_77;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(c0_108), "f"(dyh_106[0]), "f"(rh1_115));
                    float rh2_117 = _fma_77;
                    float _fma_78;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(c1_109), "f"(dyl_105[1]), "f"(rl2_116));
                    float rl3_118 = _fma_78;
                    float _fma_79;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(c1_109), "f"(dyh_106[1]), "f"(rh2_117));
                    float rh3_119 = _fma_79;
                    float _fma_80;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(c2_110), "f"(dyl_105[2]), "f"(rl3_118));
                    float rl4_120 = _fma_80;
                    float _fma_81;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(c2_110), "f"(dyh_106[2]), "f"(rh3_119));
                    float rh4_121 = _fma_81;
                    float _fma_82;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(c3_111), "f"(dyl_105[3]), "f"(rl4_120));
                    float rl5_122 = _fma_82;
                    float _fma_83;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(c3_111), "f"(dyh_106[3]), "f"(rh4_121));
                    float rh5_123 = _fma_83;
                    __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(rl5_122, rh5_123));
                    unsigned int word_124 = __as_u32(_bf16x2_6);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_62), "r"(word_124) : "memory");
                    float pre_i_125 = __uint_as_float(tv[19]);
                    float c0_126 = __uint_as_float(tv[20]);
                    float c1_127 = __uint_as_float(tv[21]);
                    float c2_128 = __uint_as_float(tv[22]);
                    float c3_129 = __uint_as_float(tv[23]);
                    float _fma_84;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[6]));
                    float rl0_130 = _fma_84;
                    float _fma_85;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[7]));
                    float rh0_131 = _fma_85;
                    float _fma_86;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(pre_i_125), "f"(dul_103), "f"(rl0_130));
                    float rl1_132 = _fma_86;
                    float _fma_87;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(pre_i_125), "f"(duh_104), "f"(rh0_131));
                    float rh1_133 = _fma_87;
                    float _fma_88;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(c0_126), "f"(dyl_105[0]), "f"(rl1_132));
                    float rl2_134 = _fma_88;
                    float _fma_89;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(c0_126), "f"(dyh_106[0]), "f"(rh1_133));
                    float rh2_135 = _fma_89;
                    float _fma_90;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(c1_127), "f"(dyl_105[1]), "f"(rl2_134));
                    float rl3_136 = _fma_90;
                    float _fma_91;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(c1_127), "f"(dyh_106[1]), "f"(rh2_135));
                    float rh3_137 = _fma_91;
                    float _fma_92;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(c2_128), "f"(dyl_105[2]), "f"(rl3_136));
                    float rl4_138 = _fma_92;
                    float _fma_93;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(c2_128), "f"(dyh_106[2]), "f"(rh3_137));
                    float rh4_139 = _fma_93;
                    float _fma_94;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(c3_129), "f"(dyl_105[3]), "f"(rl4_138));
                    float rl5_140 = _fma_94;
                    float _fma_95;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(c3_129), "f"(dyh_106[3]), "f"(rh4_139));
                    float rh5_141 = _fma_95;
                    __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(rl5_140, rh5_141));
                    unsigned int word_142 = __as_u32(_bf16x2_7);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_62 + 8192), "r"(word_142) : "memory");
                    unsigned int rb0_143 = sbase + offg[2] + rowb0;
                    unsigned int rb1_144 = rb0_143 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_143 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_144 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_143 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_144 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_143 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_144 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_143 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_144 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_143 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_144 + 57344));
                    unsigned int xw0_145 = rb0_143 + xoff0;
                    unsigned int xw1_146 = rb1_144 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_145));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_146));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_145 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_146 + 8192));
                    float dul_147 = __uint_as_float(wv0[0] << 16);
                    float duh_148 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_149[4];
                    float dyh_150[4];
                    dyl_149[0] = __uint_as_float(wv0[1] << 16);
                    dyh_150[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_149[1] = __uint_as_float(wv0[2] << 16);
                    dyh_150[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_149[2] = __uint_as_float(wv0[3] << 16);
                    dyh_150[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_149[3] = __uint_as_float(wv0[4] << 16);
                    dyh_150[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_151 = __uint_as_float(tv[1]);
                    float c0_152 = __uint_as_float(tv[2]);
                    float c1_153 = __uint_as_float(tv[3]);
                    float c2_154 = __uint_as_float(tv[4]);
                    float c3_155 = __uint_as_float(tv[5]);
                    float _fma_96;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[8]));
                    float rl0_156 = _fma_96;
                    float _fma_97;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[9]));
                    float rh0_157 = _fma_97;
                    float _fma_98;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(pre_i_151), "f"(dul_147), "f"(rl0_156));
                    float rl1_158 = _fma_98;
                    float _fma_99;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(pre_i_151), "f"(duh_148), "f"(rh0_157));
                    float rh1_159 = _fma_99;
                    float _fma_100;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(c0_152), "f"(dyl_149[0]), "f"(rl1_158));
                    float rl2_160 = _fma_100;
                    float _fma_101;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(c0_152), "f"(dyh_150[0]), "f"(rh1_159));
                    float rh2_161 = _fma_101;
                    float _fma_102;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(c1_153), "f"(dyl_149[1]), "f"(rl2_160));
                    float rl3_162 = _fma_102;
                    float _fma_103;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(c1_153), "f"(dyh_150[1]), "f"(rh2_161));
                    float rh3_163 = _fma_103;
                    float _fma_104;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(c2_154), "f"(dyl_149[2]), "f"(rl3_162));
                    float rl4_164 = _fma_104;
                    float _fma_105;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(c2_154), "f"(dyh_150[2]), "f"(rh3_163));
                    float rh4_165 = _fma_105;
                    float _fma_106;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(c3_155), "f"(dyl_149[3]), "f"(rl4_164));
                    float rl5_166 = _fma_106;
                    float _fma_107;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(c3_155), "f"(dyh_150[3]), "f"(rh4_165));
                    float rh5_167 = _fma_107;
                    __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(rl5_166, rh5_167));
                    unsigned int word_168 = __as_u32(_bf16x2_8);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_145), "r"(word_168) : "memory");
                    float pre_i_169 = __uint_as_float(tv[7]);
                    float c0_170 = __uint_as_float(tv[8]);
                    float c1_171 = __uint_as_float(tv[9]);
                    float c2_172 = __uint_as_float(tv[10]);
                    float c3_173 = __uint_as_float(tv[11]);
                    float _fma_108;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[8]));
                    float rl0_174 = _fma_108;
                    float _fma_109;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[9]));
                    float rh0_175 = _fma_109;
                    float _fma_110;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(pre_i_169), "f"(dul_147), "f"(rl0_174));
                    float rl1_176 = _fma_110;
                    float _fma_111;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(pre_i_169), "f"(duh_148), "f"(rh0_175));
                    float rh1_177 = _fma_111;
                    float _fma_112;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(c0_170), "f"(dyl_149[0]), "f"(rl1_176));
                    float rl2_178 = _fma_112;
                    float _fma_113;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(c0_170), "f"(dyh_150[0]), "f"(rh1_177));
                    float rh2_179 = _fma_113;
                    float _fma_114;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(c1_171), "f"(dyl_149[1]), "f"(rl2_178));
                    float rl3_180 = _fma_114;
                    float _fma_115;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(c1_171), "f"(dyh_150[1]), "f"(rh2_179));
                    float rh3_181 = _fma_115;
                    float _fma_116;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(c2_172), "f"(dyl_149[2]), "f"(rl3_180));
                    float rl4_182 = _fma_116;
                    float _fma_117;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(c2_172), "f"(dyh_150[2]), "f"(rh3_181));
                    float rh4_183 = _fma_117;
                    float _fma_118;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(c3_173), "f"(dyl_149[3]), "f"(rl4_182));
                    float rl5_184 = _fma_118;
                    float _fma_119;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(c3_173), "f"(dyh_150[3]), "f"(rh4_183));
                    float rh5_185 = _fma_119;
                    __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(rl5_184, rh5_185));
                    unsigned int word_186 = __as_u32(_bf16x2_9);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_145 + 8192), "r"(word_186) : "memory");
                    float dul_187 = __uint_as_float(wv1[0] << 16);
                    float duh_188 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_189[4];
                    float dyh_190[4];
                    dyl_189[0] = __uint_as_float(wv1[1] << 16);
                    dyh_190[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_189[1] = __uint_as_float(wv1[2] << 16);
                    dyh_190[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_189[2] = __uint_as_float(wv1[3] << 16);
                    dyh_190[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_189[3] = __uint_as_float(wv1[4] << 16);
                    dyh_190[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_191 = __uint_as_float(tv[13]);
                    float c0_192 = __uint_as_float(tv[14]);
                    float c1_193 = __uint_as_float(tv[15]);
                    float c2_194 = __uint_as_float(tv[16]);
                    float c3_195 = __uint_as_float(tv[17]);
                    float _fma_120;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[10]));
                    float rl0_196 = _fma_120;
                    float _fma_121;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[11]));
                    float rh0_197 = _fma_121;
                    float _fma_122;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(pre_i_191), "f"(dul_187), "f"(rl0_196));
                    float rl1_198 = _fma_122;
                    float _fma_123;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(pre_i_191), "f"(duh_188), "f"(rh0_197));
                    float rh1_199 = _fma_123;
                    float _fma_124;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(c0_192), "f"(dyl_189[0]), "f"(rl1_198));
                    float rl2_200 = _fma_124;
                    float _fma_125;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(c0_192), "f"(dyh_190[0]), "f"(rh1_199));
                    float rh2_201 = _fma_125;
                    float _fma_126;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(c1_193), "f"(dyl_189[1]), "f"(rl2_200));
                    float rl3_202 = _fma_126;
                    float _fma_127;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(c1_193), "f"(dyh_190[1]), "f"(rh2_201));
                    float rh3_203 = _fma_127;
                    float _fma_128;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_128) : "f"(c2_194), "f"(dyl_189[2]), "f"(rl3_202));
                    float rl4_204 = _fma_128;
                    float _fma_129;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_129) : "f"(c2_194), "f"(dyh_190[2]), "f"(rh3_203));
                    float rh4_205 = _fma_129;
                    float _fma_130;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_130) : "f"(c3_195), "f"(dyl_189[3]), "f"(rl4_204));
                    float rl5_206 = _fma_130;
                    float _fma_131;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_131) : "f"(c3_195), "f"(dyh_190[3]), "f"(rh4_205));
                    float rh5_207 = _fma_131;
                    __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(rl5_206, rh5_207));
                    unsigned int word_208 = __as_u32(_bf16x2_10);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_146), "r"(word_208) : "memory");
                    float pre_i_209 = __uint_as_float(tv[19]);
                    float c0_210 = __uint_as_float(tv[20]);
                    float c1_211 = __uint_as_float(tv[21]);
                    float c2_212 = __uint_as_float(tv[22]);
                    float c3_213 = __uint_as_float(tv[23]);
                    float _fma_132;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_132) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[10]));
                    float rl0_214 = _fma_132;
                    float _fma_133;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_133) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[11]));
                    float rh0_215 = _fma_133;
                    float _fma_134;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_134) : "f"(pre_i_209), "f"(dul_187), "f"(rl0_214));
                    float rl1_216 = _fma_134;
                    float _fma_135;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_135) : "f"(pre_i_209), "f"(duh_188), "f"(rh0_215));
                    float rh1_217 = _fma_135;
                    float _fma_136;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_136) : "f"(c0_210), "f"(dyl_189[0]), "f"(rl1_216));
                    float rl2_218 = _fma_136;
                    float _fma_137;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_137) : "f"(c0_210), "f"(dyh_190[0]), "f"(rh1_217));
                    float rh2_219 = _fma_137;
                    float _fma_138;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_138) : "f"(c1_211), "f"(dyl_189[1]), "f"(rl2_218));
                    float rl3_220 = _fma_138;
                    float _fma_139;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_139) : "f"(c1_211), "f"(dyh_190[1]), "f"(rh2_219));
                    float rh3_221 = _fma_139;
                    float _fma_140;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_140) : "f"(c2_212), "f"(dyl_189[2]), "f"(rl3_220));
                    float rl4_222 = _fma_140;
                    float _fma_141;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_141) : "f"(c2_212), "f"(dyh_190[2]), "f"(rh3_221));
                    float rh4_223 = _fma_141;
                    float _fma_142;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_142) : "f"(c3_213), "f"(dyl_189[3]), "f"(rl4_222));
                    float rl5_224 = _fma_142;
                    float _fma_143;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_143) : "f"(c3_213), "f"(dyh_190[3]), "f"(rh4_223));
                    float rh5_225 = _fma_143;
                    __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(rl5_224, rh5_225));
                    unsigned int word_226 = __as_u32(_bf16x2_11);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_146 + 8192), "r"(word_226) : "memory");
                    unsigned int rb0_227 = sbase + offg[3] + rowb0;
                    unsigned int rb1_228 = rb0_227 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_227 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_228 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_227 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_228 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_227 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_228 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_227 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_228 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_227 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_228 + 57344));
                    unsigned int xw0_229 = rb0_227 + xoff0;
                    unsigned int xw1_230 = rb1_228 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_229));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_230));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_229 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_230 + 8192));
                    float dul_231 = __uint_as_float(wv0[0] << 16);
                    float duh_232 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_233[4];
                    float dyh_234[4];
                    dyl_233[0] = __uint_as_float(wv0[1] << 16);
                    dyh_234[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_233[1] = __uint_as_float(wv0[2] << 16);
                    dyh_234[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_233[2] = __uint_as_float(wv0[3] << 16);
                    dyh_234[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_233[3] = __uint_as_float(wv0[4] << 16);
                    dyh_234[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_235 = __uint_as_float(tv[1]);
                    float c0_236 = __uint_as_float(tv[2]);
                    float c1_237 = __uint_as_float(tv[3]);
                    float c2_238 = __uint_as_float(tv[4]);
                    float c3_239 = __uint_as_float(tv[5]);
                    float _fma_144;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[12]));
                    float rl0_240 = _fma_144;
                    float _fma_145;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[13]));
                    float rh0_241 = _fma_145;
                    float _fma_146;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(pre_i_235), "f"(dul_231), "f"(rl0_240));
                    float rl1_242 = _fma_146;
                    float _fma_147;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(pre_i_235), "f"(duh_232), "f"(rh0_241));
                    float rh1_243 = _fma_147;
                    float _fma_148;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(c0_236), "f"(dyl_233[0]), "f"(rl1_242));
                    float rl2_244 = _fma_148;
                    float _fma_149;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(c0_236), "f"(dyh_234[0]), "f"(rh1_243));
                    float rh2_245 = _fma_149;
                    float _fma_150;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(c1_237), "f"(dyl_233[1]), "f"(rl2_244));
                    float rl3_246 = _fma_150;
                    float _fma_151;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(c1_237), "f"(dyh_234[1]), "f"(rh2_245));
                    float rh3_247 = _fma_151;
                    float _fma_152;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(c2_238), "f"(dyl_233[2]), "f"(rl3_246));
                    float rl4_248 = _fma_152;
                    float _fma_153;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(c2_238), "f"(dyh_234[2]), "f"(rh3_247));
                    float rh4_249 = _fma_153;
                    float _fma_154;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(c3_239), "f"(dyl_233[3]), "f"(rl4_248));
                    float rl5_250 = _fma_154;
                    float _fma_155;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(c3_239), "f"(dyh_234[3]), "f"(rh4_249));
                    float rh5_251 = _fma_155;
                    __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(rl5_250, rh5_251));
                    unsigned int word_252 = __as_u32(_bf16x2_12);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_229), "r"(word_252) : "memory");
                    float pre_i_253 = __uint_as_float(tv[7]);
                    float c0_254 = __uint_as_float(tv[8]);
                    float c1_255 = __uint_as_float(tv[9]);
                    float c2_256 = __uint_as_float(tv[10]);
                    float c3_257 = __uint_as_float(tv[11]);
                    float _fma_156;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[12]));
                    float rl0_258 = _fma_156;
                    float _fma_157;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[13]));
                    float rh0_259 = _fma_157;
                    float _fma_158;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(pre_i_253), "f"(dul_231), "f"(rl0_258));
                    float rl1_260 = _fma_158;
                    float _fma_159;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(pre_i_253), "f"(duh_232), "f"(rh0_259));
                    float rh1_261 = _fma_159;
                    float _fma_160;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(c0_254), "f"(dyl_233[0]), "f"(rl1_260));
                    float rl2_262 = _fma_160;
                    float _fma_161;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(c0_254), "f"(dyh_234[0]), "f"(rh1_261));
                    float rh2_263 = _fma_161;
                    float _fma_162;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(c1_255), "f"(dyl_233[1]), "f"(rl2_262));
                    float rl3_264 = _fma_162;
                    float _fma_163;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(c1_255), "f"(dyh_234[1]), "f"(rh2_263));
                    float rh3_265 = _fma_163;
                    float _fma_164;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(c2_256), "f"(dyl_233[2]), "f"(rl3_264));
                    float rl4_266 = _fma_164;
                    float _fma_165;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(c2_256), "f"(dyh_234[2]), "f"(rh3_265));
                    float rh4_267 = _fma_165;
                    float _fma_166;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(c3_257), "f"(dyl_233[3]), "f"(rl4_266));
                    float rl5_268 = _fma_166;
                    float _fma_167;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(c3_257), "f"(dyh_234[3]), "f"(rh4_267));
                    float rh5_269 = _fma_167;
                    __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(rl5_268, rh5_269));
                    unsigned int word_270 = __as_u32(_bf16x2_13);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_229 + 8192), "r"(word_270) : "memory");
                    float dul_271 = __uint_as_float(wv1[0] << 16);
                    float duh_272 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_273[4];
                    float dyh_274[4];
                    dyl_273[0] = __uint_as_float(wv1[1] << 16);
                    dyh_274[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_273[1] = __uint_as_float(wv1[2] << 16);
                    dyh_274[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_273[2] = __uint_as_float(wv1[3] << 16);
                    dyh_274[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_273[3] = __uint_as_float(wv1[4] << 16);
                    dyh_274[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_275 = __uint_as_float(tv[13]);
                    float c0_276 = __uint_as_float(tv[14]);
                    float c1_277 = __uint_as_float(tv[15]);
                    float c2_278 = __uint_as_float(tv[16]);
                    float c3_279 = __uint_as_float(tv[17]);
                    float _fma_168;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[14]));
                    float rl0_280 = _fma_168;
                    float _fma_169;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[15]));
                    float rh0_281 = _fma_169;
                    float _fma_170;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(pre_i_275), "f"(dul_271), "f"(rl0_280));
                    float rl1_282 = _fma_170;
                    float _fma_171;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(pre_i_275), "f"(duh_272), "f"(rh0_281));
                    float rh1_283 = _fma_171;
                    float _fma_172;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_172) : "f"(c0_276), "f"(dyl_273[0]), "f"(rl1_282));
                    float rl2_284 = _fma_172;
                    float _fma_173;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_173) : "f"(c0_276), "f"(dyh_274[0]), "f"(rh1_283));
                    float rh2_285 = _fma_173;
                    float _fma_174;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_174) : "f"(c1_277), "f"(dyl_273[1]), "f"(rl2_284));
                    float rl3_286 = _fma_174;
                    float _fma_175;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_175) : "f"(c1_277), "f"(dyh_274[1]), "f"(rh2_285));
                    float rh3_287 = _fma_175;
                    float _fma_176;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_176) : "f"(c2_278), "f"(dyl_273[2]), "f"(rl3_286));
                    float rl4_288 = _fma_176;
                    float _fma_177;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_177) : "f"(c2_278), "f"(dyh_274[2]), "f"(rh3_287));
                    float rh4_289 = _fma_177;
                    float _fma_178;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(c3_279), "f"(dyl_273[3]), "f"(rl4_288));
                    float rl5_290 = _fma_178;
                    float _fma_179;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(c3_279), "f"(dyh_274[3]), "f"(rh4_289));
                    float rh5_291 = _fma_179;
                    __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(rl5_290, rh5_291));
                    unsigned int word_292 = __as_u32(_bf16x2_14);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_230), "r"(word_292) : "memory");
                    float pre_i_293 = __uint_as_float(tv[19]);
                    float c0_294 = __uint_as_float(tv[20]);
                    float c1_295 = __uint_as_float(tv[21]);
                    float c2_296 = __uint_as_float(tv[22]);
                    float c3_297 = __uint_as_float(tv[23]);
                    float _fma_180;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_180) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[14]));
                    float rl0_298 = _fma_180;
                    float _fma_181;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_181) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[15]));
                    float rh0_299 = _fma_181;
                    float _fma_182;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_182) : "f"(pre_i_293), "f"(dul_271), "f"(rl0_298));
                    float rl1_300 = _fma_182;
                    float _fma_183;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_183) : "f"(pre_i_293), "f"(duh_272), "f"(rh0_299));
                    float rh1_301 = _fma_183;
                    float _fma_184;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_184) : "f"(c0_294), "f"(dyl_273[0]), "f"(rl1_300));
                    float rl2_302 = _fma_184;
                    float _fma_185;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_185) : "f"(c0_294), "f"(dyh_274[0]), "f"(rh1_301));
                    float rh2_303 = _fma_185;
                    float _fma_186;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_186) : "f"(c1_295), "f"(dyl_273[1]), "f"(rl2_302));
                    float rl3_304 = _fma_186;
                    float _fma_187;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_187) : "f"(c1_295), "f"(dyh_274[1]), "f"(rh2_303));
                    float rh3_305 = _fma_187;
                    float _fma_188;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_188) : "f"(c2_296), "f"(dyl_273[2]), "f"(rl3_304));
                    float rl4_306 = _fma_188;
                    float _fma_189;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(c2_296), "f"(dyh_274[2]), "f"(rh3_305));
                    float rh4_307 = _fma_189;
                    float _fma_190;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(c3_297), "f"(dyl_273[3]), "f"(rl4_306));
                    float rl5_308 = _fma_190;
                    float _fma_191;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_191) : "f"(c3_297), "f"(dyh_274[3]), "f"(rh4_307));
                    float rh5_309 = _fma_191;
                    __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(rl5_308, rh5_309));
                    unsigned int word_310 = __as_u32(_bf16x2_15);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_230 + 8192), "r"(word_310) : "memory");
                    float _tmem_load_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                        : "r"(tmw + tmc + 32));
                    float _tmem_load_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                        : "r"(tmw + tmc + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int rb0_311 = sbase + offg[4] + rowb0;
                    unsigned int rb1_312 = rb0_311 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_311 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_312 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_311 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_312 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_311 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_312 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_311 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_312 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_311 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_312 + 57344));
                    unsigned int xw0_313 = rb0_311 + xoff0;
                    unsigned int xw1_314 = rb1_312 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_313));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_314));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_313 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_314 + 8192));
                    float dul_315 = __uint_as_float(wv0[0] << 16);
                    float duh_316 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_317[4];
                    float dyh_318[4];
                    dyl_317[0] = __uint_as_float(wv0[1] << 16);
                    dyh_318[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_317[1] = __uint_as_float(wv0[2] << 16);
                    dyh_318[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_317[2] = __uint_as_float(wv0[3] << 16);
                    dyh_318[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_317[3] = __uint_as_float(wv0[4] << 16);
                    dyh_318[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_319 = __uint_as_float(tv[1]);
                    float c0_320 = __uint_as_float(tv[2]);
                    float c1_321 = __uint_as_float(tv[3]);
                    float c2_322 = __uint_as_float(tv[4]);
                    float c3_323 = __uint_as_float(tv[5]);
                    float _fma_192;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_192) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[0]));
                    float rl0_324 = _fma_192;
                    float _fma_193;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_193) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[1]));
                    float rh0_325 = _fma_193;
                    float _fma_194;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_194) : "f"(pre_i_319), "f"(dul_315), "f"(rl0_324));
                    float rl1_326 = _fma_194;
                    float _fma_195;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_195) : "f"(pre_i_319), "f"(duh_316), "f"(rh0_325));
                    float rh1_327 = _fma_195;
                    float _fma_196;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_196) : "f"(c0_320), "f"(dyl_317[0]), "f"(rl1_326));
                    float rl2_328 = _fma_196;
                    float _fma_197;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_197) : "f"(c0_320), "f"(dyh_318[0]), "f"(rh1_327));
                    float rh2_329 = _fma_197;
                    float _fma_198;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_198) : "f"(c1_321), "f"(dyl_317[1]), "f"(rl2_328));
                    float rl3_330 = _fma_198;
                    float _fma_199;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_199) : "f"(c1_321), "f"(dyh_318[1]), "f"(rh2_329));
                    float rh3_331 = _fma_199;
                    float _fma_200;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_200) : "f"(c2_322), "f"(dyl_317[2]), "f"(rl3_330));
                    float rl4_332 = _fma_200;
                    float _fma_201;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_201) : "f"(c2_322), "f"(dyh_318[2]), "f"(rh3_331));
                    float rh4_333 = _fma_201;
                    float _fma_202;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_202) : "f"(c3_323), "f"(dyl_317[3]), "f"(rl4_332));
                    float rl5_334 = _fma_202;
                    float _fma_203;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_203) : "f"(c3_323), "f"(dyh_318[3]), "f"(rh4_333));
                    float rh5_335 = _fma_203;
                    __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(rl5_334, rh5_335));
                    unsigned int word_336 = __as_u32(_bf16x2_16);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_313), "r"(word_336) : "memory");
                    float pre_i_337 = __uint_as_float(tv[7]);
                    float c0_338 = __uint_as_float(tv[8]);
                    float c1_339 = __uint_as_float(tv[9]);
                    float c2_340 = __uint_as_float(tv[10]);
                    float c3_341 = __uint_as_float(tv[11]);
                    float _fma_204;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_204) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[0]));
                    float rl0_342 = _fma_204;
                    float _fma_205;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_205) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[1]));
                    float rh0_343 = _fma_205;
                    float _fma_206;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_206) : "f"(pre_i_337), "f"(dul_315), "f"(rl0_342));
                    float rl1_344 = _fma_206;
                    float _fma_207;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_207) : "f"(pre_i_337), "f"(duh_316), "f"(rh0_343));
                    float rh1_345 = _fma_207;
                    float _fma_208;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_208) : "f"(c0_338), "f"(dyl_317[0]), "f"(rl1_344));
                    float rl2_346 = _fma_208;
                    float _fma_209;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_209) : "f"(c0_338), "f"(dyh_318[0]), "f"(rh1_345));
                    float rh2_347 = _fma_209;
                    float _fma_210;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_210) : "f"(c1_339), "f"(dyl_317[1]), "f"(rl2_346));
                    float rl3_348 = _fma_210;
                    float _fma_211;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_211) : "f"(c1_339), "f"(dyh_318[1]), "f"(rh2_347));
                    float rh3_349 = _fma_211;
                    float _fma_212;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_212) : "f"(c2_340), "f"(dyl_317[2]), "f"(rl3_348));
                    float rl4_350 = _fma_212;
                    float _fma_213;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_213) : "f"(c2_340), "f"(dyh_318[2]), "f"(rh3_349));
                    float rh4_351 = _fma_213;
                    float _fma_214;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_214) : "f"(c3_341), "f"(dyl_317[3]), "f"(rl4_350));
                    float rl5_352 = _fma_214;
                    float _fma_215;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_215) : "f"(c3_341), "f"(dyh_318[3]), "f"(rh4_351));
                    float rh5_353 = _fma_215;
                    __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(rl5_352, rh5_353));
                    unsigned int word_354 = __as_u32(_bf16x2_17);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_313 + 8192), "r"(word_354) : "memory");
                    float dul_355 = __uint_as_float(wv1[0] << 16);
                    float duh_356 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_357[4];
                    float dyh_358[4];
                    dyl_357[0] = __uint_as_float(wv1[1] << 16);
                    dyh_358[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_357[1] = __uint_as_float(wv1[2] << 16);
                    dyh_358[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_357[2] = __uint_as_float(wv1[3] << 16);
                    dyh_358[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_357[3] = __uint_as_float(wv1[4] << 16);
                    dyh_358[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_359 = __uint_as_float(tv[13]);
                    float c0_360 = __uint_as_float(tv[14]);
                    float c1_361 = __uint_as_float(tv[15]);
                    float c2_362 = __uint_as_float(tv[16]);
                    float c3_363 = __uint_as_float(tv[17]);
                    float _fma_216;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[2]));
                    float rl0_364 = _fma_216;
                    float _fma_217;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[3]));
                    float rh0_365 = _fma_217;
                    float _fma_218;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(pre_i_359), "f"(dul_355), "f"(rl0_364));
                    float rl1_366 = _fma_218;
                    float _fma_219;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(pre_i_359), "f"(duh_356), "f"(rh0_365));
                    float rh1_367 = _fma_219;
                    float _fma_220;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(c0_360), "f"(dyl_357[0]), "f"(rl1_366));
                    float rl2_368 = _fma_220;
                    float _fma_221;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(c0_360), "f"(dyh_358[0]), "f"(rh1_367));
                    float rh2_369 = _fma_221;
                    float _fma_222;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(c1_361), "f"(dyl_357[1]), "f"(rl2_368));
                    float rl3_370 = _fma_222;
                    float _fma_223;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(c1_361), "f"(dyh_358[1]), "f"(rh2_369));
                    float rh3_371 = _fma_223;
                    float _fma_224;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(c2_362), "f"(dyl_357[2]), "f"(rl3_370));
                    float rl4_372 = _fma_224;
                    float _fma_225;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(c2_362), "f"(dyh_358[2]), "f"(rh3_371));
                    float rh4_373 = _fma_225;
                    float _fma_226;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(c3_363), "f"(dyl_357[3]), "f"(rl4_372));
                    float rl5_374 = _fma_226;
                    float _fma_227;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(c3_363), "f"(dyh_358[3]), "f"(rh4_373));
                    float rh5_375 = _fma_227;
                    __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(rl5_374, rh5_375));
                    unsigned int word_376 = __as_u32(_bf16x2_18);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_314), "r"(word_376) : "memory");
                    float pre_i_377 = __uint_as_float(tv[19]);
                    float c0_378 = __uint_as_float(tv[20]);
                    float c1_379 = __uint_as_float(tv[21]);
                    float c2_380 = __uint_as_float(tv[22]);
                    float c3_381 = __uint_as_float(tv[23]);
                    float _fma_228;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[2]));
                    float rl0_382 = _fma_228;
                    float _fma_229;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[3]));
                    float rh0_383 = _fma_229;
                    float _fma_230;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(pre_i_377), "f"(dul_355), "f"(rl0_382));
                    float rl1_384 = _fma_230;
                    float _fma_231;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(pre_i_377), "f"(duh_356), "f"(rh0_383));
                    float rh1_385 = _fma_231;
                    float _fma_232;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(c0_378), "f"(dyl_357[0]), "f"(rl1_384));
                    float rl2_386 = _fma_232;
                    float _fma_233;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(c0_378), "f"(dyh_358[0]), "f"(rh1_385));
                    float rh2_387 = _fma_233;
                    float _fma_234;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(c1_379), "f"(dyl_357[1]), "f"(rl2_386));
                    float rl3_388 = _fma_234;
                    float _fma_235;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(c1_379), "f"(dyh_358[1]), "f"(rh2_387));
                    float rh3_389 = _fma_235;
                    float _fma_236;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(c2_380), "f"(dyl_357[2]), "f"(rl3_388));
                    float rl4_390 = _fma_236;
                    float _fma_237;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(c2_380), "f"(dyh_358[2]), "f"(rh3_389));
                    float rh4_391 = _fma_237;
                    float _fma_238;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(c3_381), "f"(dyl_357[3]), "f"(rl4_390));
                    float rl5_392 = _fma_238;
                    float _fma_239;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(c3_381), "f"(dyh_358[3]), "f"(rh4_391));
                    float rh5_393 = _fma_239;
                    __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(rl5_392, rh5_393));
                    unsigned int word_394 = __as_u32(_bf16x2_19);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_314 + 8192), "r"(word_394) : "memory");
                    unsigned int rb0_395 = sbase + offg[5] + rowb0;
                    unsigned int rb1_396 = rb0_395 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_395 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_396 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_395 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_396 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_395 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_396 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_395 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_396 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_395 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_396 + 57344));
                    unsigned int xw0_397 = rb0_395 + xoff0;
                    unsigned int xw1_398 = rb1_396 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_397));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_398));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_397 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_398 + 8192));
                    float dul_399 = __uint_as_float(wv0[0] << 16);
                    float duh_400 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_401[4];
                    float dyh_402[4];
                    dyl_401[0] = __uint_as_float(wv0[1] << 16);
                    dyh_402[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_401[1] = __uint_as_float(wv0[2] << 16);
                    dyh_402[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_401[2] = __uint_as_float(wv0[3] << 16);
                    dyh_402[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_401[3] = __uint_as_float(wv0[4] << 16);
                    dyh_402[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_403 = __uint_as_float(tv[1]);
                    float c0_404 = __uint_as_float(tv[2]);
                    float c1_405 = __uint_as_float(tv[3]);
                    float c2_406 = __uint_as_float(tv[4]);
                    float c3_407 = __uint_as_float(tv[5]);
                    float _fma_240;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[4]));
                    float rl0_408 = _fma_240;
                    float _fma_241;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[5]));
                    float rh0_409 = _fma_241;
                    float _fma_242;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(pre_i_403), "f"(dul_399), "f"(rl0_408));
                    float rl1_410 = _fma_242;
                    float _fma_243;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(pre_i_403), "f"(duh_400), "f"(rh0_409));
                    float rh1_411 = _fma_243;
                    float _fma_244;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_244) : "f"(c0_404), "f"(dyl_401[0]), "f"(rl1_410));
                    float rl2_412 = _fma_244;
                    float _fma_245;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_245) : "f"(c0_404), "f"(dyh_402[0]), "f"(rh1_411));
                    float rh2_413 = _fma_245;
                    float _fma_246;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_246) : "f"(c1_405), "f"(dyl_401[1]), "f"(rl2_412));
                    float rl3_414 = _fma_246;
                    float _fma_247;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_247) : "f"(c1_405), "f"(dyh_402[1]), "f"(rh2_413));
                    float rh3_415 = _fma_247;
                    float _fma_248;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_248) : "f"(c2_406), "f"(dyl_401[2]), "f"(rl3_414));
                    float rl4_416 = _fma_248;
                    float _fma_249;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_249) : "f"(c2_406), "f"(dyh_402[2]), "f"(rh3_415));
                    float rh4_417 = _fma_249;
                    float _fma_250;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_250) : "f"(c3_407), "f"(dyl_401[3]), "f"(rl4_416));
                    float rl5_418 = _fma_250;
                    float _fma_251;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_251) : "f"(c3_407), "f"(dyh_402[3]), "f"(rh4_417));
                    float rh5_419 = _fma_251;
                    __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(rl5_418, rh5_419));
                    unsigned int word_420 = __as_u32(_bf16x2_20);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_397), "r"(word_420) : "memory");
                    float pre_i_421 = __uint_as_float(tv[7]);
                    float c0_422 = __uint_as_float(tv[8]);
                    float c1_423 = __uint_as_float(tv[9]);
                    float c2_424 = __uint_as_float(tv[10]);
                    float c3_425 = __uint_as_float(tv[11]);
                    float _fma_252;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_252) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[4]));
                    float rl0_426 = _fma_252;
                    float _fma_253;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_253) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[5]));
                    float rh0_427 = _fma_253;
                    float _fma_254;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_254) : "f"(pre_i_421), "f"(dul_399), "f"(rl0_426));
                    float rl1_428 = _fma_254;
                    float _fma_255;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_255) : "f"(pre_i_421), "f"(duh_400), "f"(rh0_427));
                    float rh1_429 = _fma_255;
                    float _fma_256;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_256) : "f"(c0_422), "f"(dyl_401[0]), "f"(rl1_428));
                    float rl2_430 = _fma_256;
                    float _fma_257;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_257) : "f"(c0_422), "f"(dyh_402[0]), "f"(rh1_429));
                    float rh2_431 = _fma_257;
                    float _fma_258;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_258) : "f"(c1_423), "f"(dyl_401[1]), "f"(rl2_430));
                    float rl3_432 = _fma_258;
                    float _fma_259;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_259) : "f"(c1_423), "f"(dyh_402[1]), "f"(rh2_431));
                    float rh3_433 = _fma_259;
                    float _fma_260;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_260) : "f"(c2_424), "f"(dyl_401[2]), "f"(rl3_432));
                    float rl4_434 = _fma_260;
                    float _fma_261;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_261) : "f"(c2_424), "f"(dyh_402[2]), "f"(rh3_433));
                    float rh4_435 = _fma_261;
                    float _fma_262;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_262) : "f"(c3_425), "f"(dyl_401[3]), "f"(rl4_434));
                    float rl5_436 = _fma_262;
                    float _fma_263;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_263) : "f"(c3_425), "f"(dyh_402[3]), "f"(rh4_435));
                    float rh5_437 = _fma_263;
                    __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(rl5_436, rh5_437));
                    unsigned int word_438 = __as_u32(_bf16x2_21);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_397 + 8192), "r"(word_438) : "memory");
                    float dul_439 = __uint_as_float(wv1[0] << 16);
                    float duh_440 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_441[4];
                    float dyh_442[4];
                    dyl_441[0] = __uint_as_float(wv1[1] << 16);
                    dyh_442[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_441[1] = __uint_as_float(wv1[2] << 16);
                    dyh_442[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_441[2] = __uint_as_float(wv1[3] << 16);
                    dyh_442[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_441[3] = __uint_as_float(wv1[4] << 16);
                    dyh_442[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_443 = __uint_as_float(tv[13]);
                    float c0_444 = __uint_as_float(tv[14]);
                    float c1_445 = __uint_as_float(tv[15]);
                    float c2_446 = __uint_as_float(tv[16]);
                    float c3_447 = __uint_as_float(tv[17]);
                    float _fma_264;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_264) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[6]));
                    float rl0_448 = _fma_264;
                    float _fma_265;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_265) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[7]));
                    float rh0_449 = _fma_265;
                    float _fma_266;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_266) : "f"(pre_i_443), "f"(dul_439), "f"(rl0_448));
                    float rl1_450 = _fma_266;
                    float _fma_267;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_267) : "f"(pre_i_443), "f"(duh_440), "f"(rh0_449));
                    float rh1_451 = _fma_267;
                    float _fma_268;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_268) : "f"(c0_444), "f"(dyl_441[0]), "f"(rl1_450));
                    float rl2_452 = _fma_268;
                    float _fma_269;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_269) : "f"(c0_444), "f"(dyh_442[0]), "f"(rh1_451));
                    float rh2_453 = _fma_269;
                    float _fma_270;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_270) : "f"(c1_445), "f"(dyl_441[1]), "f"(rl2_452));
                    float rl3_454 = _fma_270;
                    float _fma_271;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_271) : "f"(c1_445), "f"(dyh_442[1]), "f"(rh2_453));
                    float rh3_455 = _fma_271;
                    float _fma_272;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_272) : "f"(c2_446), "f"(dyl_441[2]), "f"(rl3_454));
                    float rl4_456 = _fma_272;
                    float _fma_273;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_273) : "f"(c2_446), "f"(dyh_442[2]), "f"(rh3_455));
                    float rh4_457 = _fma_273;
                    float _fma_274;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_274) : "f"(c3_447), "f"(dyl_441[3]), "f"(rl4_456));
                    float rl5_458 = _fma_274;
                    float _fma_275;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_275) : "f"(c3_447), "f"(dyh_442[3]), "f"(rh4_457));
                    float rh5_459 = _fma_275;
                    __nv_bfloat162 _bf16x2_22 = __float22bfloat162_rn(make_float2(rl5_458, rh5_459));
                    unsigned int word_460 = __as_u32(_bf16x2_22);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_398), "r"(word_460) : "memory");
                    float pre_i_461 = __uint_as_float(tv[19]);
                    float c0_462 = __uint_as_float(tv[20]);
                    float c1_463 = __uint_as_float(tv[21]);
                    float c2_464 = __uint_as_float(tv[22]);
                    float c3_465 = __uint_as_float(tv[23]);
                    float _fma_276;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_276) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[6]));
                    float rl0_466 = _fma_276;
                    float _fma_277;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_277) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[7]));
                    float rh0_467 = _fma_277;
                    float _fma_278;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_278) : "f"(pre_i_461), "f"(dul_439), "f"(rl0_466));
                    float rl1_468 = _fma_278;
                    float _fma_279;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_279) : "f"(pre_i_461), "f"(duh_440), "f"(rh0_467));
                    float rh1_469 = _fma_279;
                    float _fma_280;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_280) : "f"(c0_462), "f"(dyl_441[0]), "f"(rl1_468));
                    float rl2_470 = _fma_280;
                    float _fma_281;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_281) : "f"(c0_462), "f"(dyh_442[0]), "f"(rh1_469));
                    float rh2_471 = _fma_281;
                    float _fma_282;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_282) : "f"(c1_463), "f"(dyl_441[1]), "f"(rl2_470));
                    float rl3_472 = _fma_282;
                    float _fma_283;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_283) : "f"(c1_463), "f"(dyh_442[1]), "f"(rh2_471));
                    float rh3_473 = _fma_283;
                    float _fma_284;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_284) : "f"(c2_464), "f"(dyl_441[2]), "f"(rl3_472));
                    float rl4_474 = _fma_284;
                    float _fma_285;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_285) : "f"(c2_464), "f"(dyh_442[2]), "f"(rh3_473));
                    float rh4_475 = _fma_285;
                    float _fma_286;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_286) : "f"(c3_465), "f"(dyl_441[3]), "f"(rl4_474));
                    float rl5_476 = _fma_286;
                    float _fma_287;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_287) : "f"(c3_465), "f"(dyh_442[3]), "f"(rh4_475));
                    float rh5_477 = _fma_287;
                    __nv_bfloat162 _bf16x2_23 = __float22bfloat162_rn(make_float2(rl5_476, rh5_477));
                    unsigned int word_478 = __as_u32(_bf16x2_23);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_398 + 8192), "r"(word_478) : "memory");
                    unsigned int rb0_479 = sbase + offg[6] + rowb0;
                    unsigned int rb1_480 = rb0_479 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_479 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_480 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_479 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_480 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_479 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_480 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_479 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_480 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_479 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_480 + 57344));
                    unsigned int xw0_481 = rb0_479 + xoff0;
                    unsigned int xw1_482 = rb1_480 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_481));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_482));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_481 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_482 + 8192));
                    float dul_483 = __uint_as_float(wv0[0] << 16);
                    float duh_484 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_485[4];
                    float dyh_486[4];
                    dyl_485[0] = __uint_as_float(wv0[1] << 16);
                    dyh_486[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_485[1] = __uint_as_float(wv0[2] << 16);
                    dyh_486[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_485[2] = __uint_as_float(wv0[3] << 16);
                    dyh_486[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_485[3] = __uint_as_float(wv0[4] << 16);
                    dyh_486[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_487 = __uint_as_float(tv[1]);
                    float c0_488 = __uint_as_float(tv[2]);
                    float c1_489 = __uint_as_float(tv[3]);
                    float c2_490 = __uint_as_float(tv[4]);
                    float c3_491 = __uint_as_float(tv[5]);
                    float _fma_288;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_288) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[8]));
                    float rl0_492 = _fma_288;
                    float _fma_289;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_289) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[9]));
                    float rh0_493 = _fma_289;
                    float _fma_290;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_290) : "f"(pre_i_487), "f"(dul_483), "f"(rl0_492));
                    float rl1_494 = _fma_290;
                    float _fma_291;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_291) : "f"(pre_i_487), "f"(duh_484), "f"(rh0_493));
                    float rh1_495 = _fma_291;
                    float _fma_292;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_292) : "f"(c0_488), "f"(dyl_485[0]), "f"(rl1_494));
                    float rl2_496 = _fma_292;
                    float _fma_293;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_293) : "f"(c0_488), "f"(dyh_486[0]), "f"(rh1_495));
                    float rh2_497 = _fma_293;
                    float _fma_294;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_294) : "f"(c1_489), "f"(dyl_485[1]), "f"(rl2_496));
                    float rl3_498 = _fma_294;
                    float _fma_295;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_295) : "f"(c1_489), "f"(dyh_486[1]), "f"(rh2_497));
                    float rh3_499 = _fma_295;
                    float _fma_296;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_296) : "f"(c2_490), "f"(dyl_485[2]), "f"(rl3_498));
                    float rl4_500 = _fma_296;
                    float _fma_297;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_297) : "f"(c2_490), "f"(dyh_486[2]), "f"(rh3_499));
                    float rh4_501 = _fma_297;
                    float _fma_298;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_298) : "f"(c3_491), "f"(dyl_485[3]), "f"(rl4_500));
                    float rl5_502 = _fma_298;
                    float _fma_299;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_299) : "f"(c3_491), "f"(dyh_486[3]), "f"(rh4_501));
                    float rh5_503 = _fma_299;
                    __nv_bfloat162 _bf16x2_24 = __float22bfloat162_rn(make_float2(rl5_502, rh5_503));
                    unsigned int word_504 = __as_u32(_bf16x2_24);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_481), "r"(word_504) : "memory");
                    float pre_i_505 = __uint_as_float(tv[7]);
                    float c0_506 = __uint_as_float(tv[8]);
                    float c1_507 = __uint_as_float(tv[9]);
                    float c2_508 = __uint_as_float(tv[10]);
                    float c3_509 = __uint_as_float(tv[11]);
                    float _fma_300;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_300) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[8]));
                    float rl0_510 = _fma_300;
                    float _fma_301;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_301) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[9]));
                    float rh0_511 = _fma_301;
                    float _fma_302;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_302) : "f"(pre_i_505), "f"(dul_483), "f"(rl0_510));
                    float rl1_512 = _fma_302;
                    float _fma_303;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_303) : "f"(pre_i_505), "f"(duh_484), "f"(rh0_511));
                    float rh1_513 = _fma_303;
                    float _fma_304;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_304) : "f"(c0_506), "f"(dyl_485[0]), "f"(rl1_512));
                    float rl2_514 = _fma_304;
                    float _fma_305;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_305) : "f"(c0_506), "f"(dyh_486[0]), "f"(rh1_513));
                    float rh2_515 = _fma_305;
                    float _fma_306;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_306) : "f"(c1_507), "f"(dyl_485[1]), "f"(rl2_514));
                    float rl3_516 = _fma_306;
                    float _fma_307;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_307) : "f"(c1_507), "f"(dyh_486[1]), "f"(rh2_515));
                    float rh3_517 = _fma_307;
                    float _fma_308;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_308) : "f"(c2_508), "f"(dyl_485[2]), "f"(rl3_516));
                    float rl4_518 = _fma_308;
                    float _fma_309;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_309) : "f"(c2_508), "f"(dyh_486[2]), "f"(rh3_517));
                    float rh4_519 = _fma_309;
                    float _fma_310;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_310) : "f"(c3_509), "f"(dyl_485[3]), "f"(rl4_518));
                    float rl5_520 = _fma_310;
                    float _fma_311;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_311) : "f"(c3_509), "f"(dyh_486[3]), "f"(rh4_519));
                    float rh5_521 = _fma_311;
                    __nv_bfloat162 _bf16x2_25 = __float22bfloat162_rn(make_float2(rl5_520, rh5_521));
                    unsigned int word_522 = __as_u32(_bf16x2_25);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_481 + 8192), "r"(word_522) : "memory");
                    float dul_523 = __uint_as_float(wv1[0] << 16);
                    float duh_524 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_525[4];
                    float dyh_526[4];
                    dyl_525[0] = __uint_as_float(wv1[1] << 16);
                    dyh_526[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_525[1] = __uint_as_float(wv1[2] << 16);
                    dyh_526[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_525[2] = __uint_as_float(wv1[3] << 16);
                    dyh_526[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_525[3] = __uint_as_float(wv1[4] << 16);
                    dyh_526[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_527 = __uint_as_float(tv[13]);
                    float c0_528 = __uint_as_float(tv[14]);
                    float c1_529 = __uint_as_float(tv[15]);
                    float c2_530 = __uint_as_float(tv[16]);
                    float c3_531 = __uint_as_float(tv[17]);
                    float _fma_312;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_312) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[10]));
                    float rl0_532 = _fma_312;
                    float _fma_313;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_313) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[11]));
                    float rh0_533 = _fma_313;
                    float _fma_314;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_314) : "f"(pre_i_527), "f"(dul_523), "f"(rl0_532));
                    float rl1_534 = _fma_314;
                    float _fma_315;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_315) : "f"(pre_i_527), "f"(duh_524), "f"(rh0_533));
                    float rh1_535 = _fma_315;
                    float _fma_316;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_316) : "f"(c0_528), "f"(dyl_525[0]), "f"(rl1_534));
                    float rl2_536 = _fma_316;
                    float _fma_317;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_317) : "f"(c0_528), "f"(dyh_526[0]), "f"(rh1_535));
                    float rh2_537 = _fma_317;
                    float _fma_318;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_318) : "f"(c1_529), "f"(dyl_525[1]), "f"(rl2_536));
                    float rl3_538 = _fma_318;
                    float _fma_319;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_319) : "f"(c1_529), "f"(dyh_526[1]), "f"(rh2_537));
                    float rh3_539 = _fma_319;
                    float _fma_320;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_320) : "f"(c2_530), "f"(dyl_525[2]), "f"(rl3_538));
                    float rl4_540 = _fma_320;
                    float _fma_321;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_321) : "f"(c2_530), "f"(dyh_526[2]), "f"(rh3_539));
                    float rh4_541 = _fma_321;
                    float _fma_322;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_322) : "f"(c3_531), "f"(dyl_525[3]), "f"(rl4_540));
                    float rl5_542 = _fma_322;
                    float _fma_323;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_323) : "f"(c3_531), "f"(dyh_526[3]), "f"(rh4_541));
                    float rh5_543 = _fma_323;
                    __nv_bfloat162 _bf16x2_26 = __float22bfloat162_rn(make_float2(rl5_542, rh5_543));
                    unsigned int word_544 = __as_u32(_bf16x2_26);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_482), "r"(word_544) : "memory");
                    float pre_i_545 = __uint_as_float(tv[19]);
                    float c0_546 = __uint_as_float(tv[20]);
                    float c1_547 = __uint_as_float(tv[21]);
                    float c2_548 = __uint_as_float(tv[22]);
                    float c3_549 = __uint_as_float(tv[23]);
                    float _fma_324;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_324) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[10]));
                    float rl0_550 = _fma_324;
                    float _fma_325;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_325) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[11]));
                    float rh0_551 = _fma_325;
                    float _fma_326;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_326) : "f"(pre_i_545), "f"(dul_523), "f"(rl0_550));
                    float rl1_552 = _fma_326;
                    float _fma_327;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_327) : "f"(pre_i_545), "f"(duh_524), "f"(rh0_551));
                    float rh1_553 = _fma_327;
                    float _fma_328;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_328) : "f"(c0_546), "f"(dyl_525[0]), "f"(rl1_552));
                    float rl2_554 = _fma_328;
                    float _fma_329;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_329) : "f"(c0_546), "f"(dyh_526[0]), "f"(rh1_553));
                    float rh2_555 = _fma_329;
                    float _fma_330;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_330) : "f"(c1_547), "f"(dyl_525[1]), "f"(rl2_554));
                    float rl3_556 = _fma_330;
                    float _fma_331;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_331) : "f"(c1_547), "f"(dyh_526[1]), "f"(rh2_555));
                    float rh3_557 = _fma_331;
                    float _fma_332;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_332) : "f"(c2_548), "f"(dyl_525[2]), "f"(rl3_556));
                    float rl4_558 = _fma_332;
                    float _fma_333;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_333) : "f"(c2_548), "f"(dyh_526[2]), "f"(rh3_557));
                    float rh4_559 = _fma_333;
                    float _fma_334;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_334) : "f"(c3_549), "f"(dyl_525[3]), "f"(rl4_558));
                    float rl5_560 = _fma_334;
                    float _fma_335;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_335) : "f"(c3_549), "f"(dyh_526[3]), "f"(rh4_559));
                    float rh5_561 = _fma_335;
                    __nv_bfloat162 _bf16x2_27 = __float22bfloat162_rn(make_float2(rl5_560, rh5_561));
                    unsigned int word_562 = __as_u32(_bf16x2_27);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_482 + 8192), "r"(word_562) : "memory");
                    unsigned int rb0_563 = sbase + offg[7] + rowb0;
                    unsigned int rb1_564 = rb0_563 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_563 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_564 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_563 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_564 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_563 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_564 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_563 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_564 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_563 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_564 + 57344));
                    unsigned int xw0_565 = rb0_563 + xoff0;
                    unsigned int xw1_566 = rb1_564 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_565));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_566));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_565 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_566 + 8192));
                    float dul_567 = __uint_as_float(wv0[0] << 16);
                    float duh_568 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_569[4];
                    float dyh_570[4];
                    dyl_569[0] = __uint_as_float(wv0[1] << 16);
                    dyh_570[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_569[1] = __uint_as_float(wv0[2] << 16);
                    dyh_570[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_569[2] = __uint_as_float(wv0[3] << 16);
                    dyh_570[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_569[3] = __uint_as_float(wv0[4] << 16);
                    dyh_570[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_571 = __uint_as_float(tv[1]);
                    float c0_572 = __uint_as_float(tv[2]);
                    float c1_573 = __uint_as_float(tv[3]);
                    float c2_574 = __uint_as_float(tv[4]);
                    float c3_575 = __uint_as_float(tv[5]);
                    float _fma_336;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_336) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[12]));
                    float rl0_576 = _fma_336;
                    float _fma_337;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_337) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[13]));
                    float rh0_577 = _fma_337;
                    float _fma_338;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_338) : "f"(pre_i_571), "f"(dul_567), "f"(rl0_576));
                    float rl1_578 = _fma_338;
                    float _fma_339;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_339) : "f"(pre_i_571), "f"(duh_568), "f"(rh0_577));
                    float rh1_579 = _fma_339;
                    float _fma_340;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_340) : "f"(c0_572), "f"(dyl_569[0]), "f"(rl1_578));
                    float rl2_580 = _fma_340;
                    float _fma_341;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_341) : "f"(c0_572), "f"(dyh_570[0]), "f"(rh1_579));
                    float rh2_581 = _fma_341;
                    float _fma_342;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_342) : "f"(c1_573), "f"(dyl_569[1]), "f"(rl2_580));
                    float rl3_582 = _fma_342;
                    float _fma_343;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_343) : "f"(c1_573), "f"(dyh_570[1]), "f"(rh2_581));
                    float rh3_583 = _fma_343;
                    float _fma_344;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_344) : "f"(c2_574), "f"(dyl_569[2]), "f"(rl3_582));
                    float rl4_584 = _fma_344;
                    float _fma_345;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_345) : "f"(c2_574), "f"(dyh_570[2]), "f"(rh3_583));
                    float rh4_585 = _fma_345;
                    float _fma_346;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_346) : "f"(c3_575), "f"(dyl_569[3]), "f"(rl4_584));
                    float rl5_586 = _fma_346;
                    float _fma_347;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_347) : "f"(c3_575), "f"(dyh_570[3]), "f"(rh4_585));
                    float rh5_587 = _fma_347;
                    __nv_bfloat162 _bf16x2_28 = __float22bfloat162_rn(make_float2(rl5_586, rh5_587));
                    unsigned int word_588 = __as_u32(_bf16x2_28);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_565), "r"(word_588) : "memory");
                    float pre_i_589 = __uint_as_float(tv[7]);
                    float c0_590 = __uint_as_float(tv[8]);
                    float c1_591 = __uint_as_float(tv[9]);
                    float c2_592 = __uint_as_float(tv[10]);
                    float c3_593 = __uint_as_float(tv[11]);
                    float _fma_348;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_348) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[12]));
                    float rl0_594 = _fma_348;
                    float _fma_349;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_349) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[13]));
                    float rh0_595 = _fma_349;
                    float _fma_350;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_350) : "f"(pre_i_589), "f"(dul_567), "f"(rl0_594));
                    float rl1_596 = _fma_350;
                    float _fma_351;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_351) : "f"(pre_i_589), "f"(duh_568), "f"(rh0_595));
                    float rh1_597 = _fma_351;
                    float _fma_352;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_352) : "f"(c0_590), "f"(dyl_569[0]), "f"(rl1_596));
                    float rl2_598 = _fma_352;
                    float _fma_353;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_353) : "f"(c0_590), "f"(dyh_570[0]), "f"(rh1_597));
                    float rh2_599 = _fma_353;
                    float _fma_354;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_354) : "f"(c1_591), "f"(dyl_569[1]), "f"(rl2_598));
                    float rl3_600 = _fma_354;
                    float _fma_355;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_355) : "f"(c1_591), "f"(dyh_570[1]), "f"(rh2_599));
                    float rh3_601 = _fma_355;
                    float _fma_356;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_356) : "f"(c2_592), "f"(dyl_569[2]), "f"(rl3_600));
                    float rl4_602 = _fma_356;
                    float _fma_357;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_357) : "f"(c2_592), "f"(dyh_570[2]), "f"(rh3_601));
                    float rh4_603 = _fma_357;
                    float _fma_358;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_358) : "f"(c3_593), "f"(dyl_569[3]), "f"(rl4_602));
                    float rl5_604 = _fma_358;
                    float _fma_359;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_359) : "f"(c3_593), "f"(dyh_570[3]), "f"(rh4_603));
                    float rh5_605 = _fma_359;
                    __nv_bfloat162 _bf16x2_29 = __float22bfloat162_rn(make_float2(rl5_604, rh5_605));
                    unsigned int word_606 = __as_u32(_bf16x2_29);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_565 + 8192), "r"(word_606) : "memory");
                    float dul_607 = __uint_as_float(wv1[0] << 16);
                    float duh_608 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_609[4];
                    float dyh_610[4];
                    dyl_609[0] = __uint_as_float(wv1[1] << 16);
                    dyh_610[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_609[1] = __uint_as_float(wv1[2] << 16);
                    dyh_610[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_609[2] = __uint_as_float(wv1[3] << 16);
                    dyh_610[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_609[3] = __uint_as_float(wv1[4] << 16);
                    dyh_610[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_611 = __uint_as_float(tv[13]);
                    float c0_612 = __uint_as_float(tv[14]);
                    float c1_613 = __uint_as_float(tv[15]);
                    float c2_614 = __uint_as_float(tv[16]);
                    float c3_615 = __uint_as_float(tv[17]);
                    float _fma_360;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_360) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[14]));
                    float rl0_616 = _fma_360;
                    float _fma_361;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_361) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[15]));
                    float rh0_617 = _fma_361;
                    float _fma_362;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_362) : "f"(pre_i_611), "f"(dul_607), "f"(rl0_616));
                    float rl1_618 = _fma_362;
                    float _fma_363;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_363) : "f"(pre_i_611), "f"(duh_608), "f"(rh0_617));
                    float rh1_619 = _fma_363;
                    float _fma_364;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_364) : "f"(c0_612), "f"(dyl_609[0]), "f"(rl1_618));
                    float rl2_620 = _fma_364;
                    float _fma_365;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_365) : "f"(c0_612), "f"(dyh_610[0]), "f"(rh1_619));
                    float rh2_621 = _fma_365;
                    float _fma_366;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_366) : "f"(c1_613), "f"(dyl_609[1]), "f"(rl2_620));
                    float rl3_622 = _fma_366;
                    float _fma_367;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_367) : "f"(c1_613), "f"(dyh_610[1]), "f"(rh2_621));
                    float rh3_623 = _fma_367;
                    float _fma_368;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_368) : "f"(c2_614), "f"(dyl_609[2]), "f"(rl3_622));
                    float rl4_624 = _fma_368;
                    float _fma_369;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_369) : "f"(c2_614), "f"(dyh_610[2]), "f"(rh3_623));
                    float rh4_625 = _fma_369;
                    float _fma_370;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_370) : "f"(c3_615), "f"(dyl_609[3]), "f"(rl4_624));
                    float rl5_626 = _fma_370;
                    float _fma_371;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_371) : "f"(c3_615), "f"(dyh_610[3]), "f"(rh4_625));
                    float rh5_627 = _fma_371;
                    __nv_bfloat162 _bf16x2_30 = __float22bfloat162_rn(make_float2(rl5_626, rh5_627));
                    unsigned int word_628 = __as_u32(_bf16x2_30);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_566), "r"(word_628) : "memory");
                    float pre_i_629 = __uint_as_float(tv[19]);
                    float c0_630 = __uint_as_float(tv[20]);
                    float c1_631 = __uint_as_float(tv[21]);
                    float c2_632 = __uint_as_float(tv[22]);
                    float c3_633 = __uint_as_float(tv[23]);
                    float _fma_372;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_372) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[14]));
                    float rl0_634 = _fma_372;
                    float _fma_373;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_373) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[15]));
                    float rh0_635 = _fma_373;
                    float _fma_374;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_374) : "f"(pre_i_629), "f"(dul_607), "f"(rl0_634));
                    float rl1_636 = _fma_374;
                    float _fma_375;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_375) : "f"(pre_i_629), "f"(duh_608), "f"(rh0_635));
                    float rh1_637 = _fma_375;
                    float _fma_376;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_376) : "f"(c0_630), "f"(dyl_609[0]), "f"(rl1_636));
                    float rl2_638 = _fma_376;
                    float _fma_377;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_377) : "f"(c0_630), "f"(dyh_610[0]), "f"(rh1_637));
                    float rh2_639 = _fma_377;
                    float _fma_378;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_378) : "f"(c1_631), "f"(dyl_609[1]), "f"(rl2_638));
                    float rl3_640 = _fma_378;
                    float _fma_379;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_379) : "f"(c1_631), "f"(dyh_610[1]), "f"(rh2_639));
                    float rh3_641 = _fma_379;
                    float _fma_380;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_380) : "f"(c2_632), "f"(dyl_609[2]), "f"(rl3_640));
                    float rl4_642 = _fma_380;
                    float _fma_381;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_381) : "f"(c2_632), "f"(dyh_610[2]), "f"(rh3_641));
                    float rh4_643 = _fma_381;
                    float _fma_382;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_382) : "f"(c3_633), "f"(dyl_609[3]), "f"(rl4_642));
                    float rl5_644 = _fma_382;
                    float _fma_383;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_383) : "f"(c3_633), "f"(dyh_610[3]), "f"(rh4_643));
                    float rh5_645 = _fma_383;
                    __nv_bfloat162 _bf16x2_31 = __float22bfloat162_rn(make_float2(rl5_644, rh5_645));
                    unsigned int word_646 = __as_u32(_bf16x2_31);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_566 + 8192), "r"(word_646) : "memory");
                    int nxt = tile_2 + 1;
                    if (nxt < n_tiles_2) {
                        if (etid < 192) {
                            int rowa_1 = t0_1 + 64 + fr;
                            if (rowa_1 >= T) {
                                rowa_1 = T - 1;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(tokp + (unsigned long long)rowa_1 * 24 + fq);
                                qa[0 + 0] = _v4.x;
                                qa[0 + 1] = _v4.y;
                                qa[0 + 2] = _v4.z;
                                qa[0 + 3] = _v4.w;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(tokp + (unsigned long long)rowa_1 * 24 + fq + 4);
                                qa[4 + 0] = _v4.x;
                                qa[4 + 1] = _v4.y;
                                qa[4 + 2] = _v4.z;
                                qa[4 + 3] = _v4.w;
                            }
                        }
                    }
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(lane_tb + acc_ep * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int n_1 = 0; n_1 < 24; n_1++) {
                        float _sub_0;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_tmem_load_4[n_1]), "f"(comp[n_1]));
                        float _add_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(acc[n_1]), "f"(_sub_0));
                        float _sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(_add_0), "f"(acc[n_1]));
                        float _sub_2;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_sub_1), "f"(_sub_0));
                        comp[n_1] = _sub_2;
                        acc[n_1] = _add_0;
                    }
                    int in_grp_1 = tile_2 % 16;
                    int prom = 0;
                    if (in_grp_1 == 15) {
                        prom = 1;
                    }
                    if (tile_2 == n_tiles_2 - 1) {
                        prom = 1;
                    }
                    if (prom == 1) {
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                            : "r"(lane_ts + grp_ep % 2 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int n_2 = 0; n_2 < 24; n_2++) {
                            float _sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_tmem_load_5[n_2]), "f"(comp[n_2]));
                            float _add_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(acc[n_2]), "f"(_sub_3));
                            float _sub_4;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(_add_1), "f"(acc[n_2]));
                            float _sub_5;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(_sub_4), "f"(_sub_3));
                            comp[n_2] = _sub_5;
                            acc[n_2] = _add_1;
                        }
                        grp_ep = grp_ep + 1;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(acc_empty_addr + (acc_ep) * 8);
                    }
                    acc_ep += 1;
                    if (acc_ep == 2) { acc_ep = 0; _phase_acc_full ^= 1; }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                    if (ew == 0) {
                        if (_elect_sync_0) {
                            tma_store_2d((&dX), s_3 * 64, t0_1, x0_addr + ep_stage * 86016);
                            tma_store_2d((&dX), s_3 * 64 + 4096, t0_1, x1_addr + ep_stage * 86016);
                            tma_store_2d((&dX), s_3 * 64 + 8192, t0_1, x2_addr + ep_stage * 86016);
                            tma_store_2d((&dX), s_3 * 64 + 12288, t0_1, x3_addr + ep_stage * 86016);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                        pend = 1;
                        pend_stage = ep_stage;
                    }
                    ep_stage += 1;
                    if (ep_stage == 2) { ep_stage = 0; _phase_stage_full_1 ^= 1; }
                }
                unsigned long long kcol = (unsigned long long)(i_str * 4096 + s_3 * 64 + dl);
                unsigned long long pbase = (unsigned long long)p_3 * 393216;
                #pragma unroll
                for (int n_3 = 0; n_3 < 24; n_3++) {
                    float _sub_6;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(acc[n_3]), "f"(comp[n_3]));
                    dfn_part[pbase + (unsigned long long)(n_3 * 16384) + kcol] = _sub_6;
                }
            }
            if (ew == 0) {
                if (pend == 1) {
                    if (_elect_sync_0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(stage_empty_addr + (pend_stage) * 8);
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
