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
#define SMEM_WTV_OFF 222208
#define SMEM_WTV_STAGE_BYTES 10240
#define SMEM_WTV_STRIDE 10240
#define SMEM_X0_1_0_OFF 1024
#define SMEM_X0_1_0_STAGE_BYTES 8192
#define SMEM_X0_1_0_STRIDE 86016
#define SMEM_DY0_1_0_OFF 33792
#define SMEM_DY0_1_0_STAGE_BYTES 8192
#define SMEM_DY0_1_0_STRIDE 86016
#define SMEM_X0_1_1_OFF 9216
#define SMEM_X0_1_1_STAGE_BYTES 8192
#define SMEM_X0_1_1_STRIDE 86016
#define SMEM_DY0_1_1_OFF 41984
#define SMEM_DY0_1_1_STAGE_BYTES 8192
#define SMEM_DY0_1_1_STRIDE 86016
#define SMEM_X0_1_2_OFF 17408
#define SMEM_X0_1_2_STAGE_BYTES 8192
#define SMEM_X0_1_2_STRIDE 86016
#define SMEM_DY0_1_2_OFF 50176
#define SMEM_DY0_1_2_STAGE_BYTES 8192
#define SMEM_DY0_1_2_STRIDE 86016
#define SMEM_X0_1_3_OFF 25600
#define SMEM_X0_1_3_STAGE_BYTES 8192
#define SMEM_X0_1_3_STRIDE 86016
#define SMEM_DY0_1_3_OFF 58368
#define SMEM_DY0_1_3_STAGE_BYTES 8192
#define SMEM_DY0_1_3_STRIDE 86016
#define SMEM_DUS_OFF 66560
#define SMEM_DUS_STAGE_BYTES 8192
#define SMEM_DUS_STRIDE 86016
#define SMEM_DZ0_1_0_OFF 74752
#define SMEM_DZ0_1_0_STAGE_BYTES 4096
#define SMEM_DZ0_1_0_STRIDE 86016
#define SMEM_DZ0_1_1_OFF 78848
#define SMEM_DZ0_1_1_STAGE_BYTES 4096
#define SMEM_DZ0_1_1_STRIDE 86016
#define SMEM_DZ0_1_2_OFF 82944
#define SMEM_DZ0_1_2_STAGE_BYTES 4096
#define SMEM_DZ0_1_2_STRIDE 86016
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
#define SMEM_FN2_2_OFF 205824
#define SMEM_FN2_2_STAGE_BYTES 8192
#define SMEM_FN2_2_STRIDE 8192
#define SMEM_FN2_3_OFF 214016
#define SMEM_FN2_3_STAGE_BYTES 8192
#define SMEM_FN2_3_STRIDE 8192
#define SMEM_FNW0_1_0_OFF 173056
#define SMEM_FNW0_1_0_STAGE_BYTES 16384
#define SMEM_FNW0_1_0_STRIDE 16384
#define SMEM_FNW0_1_1_OFF 189440
#define SMEM_FNW0_1_1_STAGE_BYTES 16384
#define SMEM_FNW0_1_1_STRIDE 16384
#define SMEM_FNW0_1_2_OFF 205824
#define SMEM_FNW0_1_2_STAGE_BYTES 16384
#define SMEM_FNW0_1_2_STRIDE 16384
#define SMEM_XA0_1_0_OFF 1024
#define SMEM_XA0_1_0_STAGE_BYTES 16384
#define SMEM_XA0_1_0_STRIDE 86016
#define SMEM_XA0_1_1_OFF 17408
#define SMEM_XA0_1_1_STAGE_BYTES 16384
#define SMEM_XA0_1_1_STRIDE 86016
#define SMEM_DZN0_OFF 74752
#define SMEM_DZN0_STAGE_BYTES 4096
#define SMEM_DZN0_STRIDE 86016
#define SMEM_DZN1_OFF 78848
#define SMEM_DZN1_STAGE_BYTES 4096
#define SMEM_DZN1_STRIDE 86016
#define SMEM_DZN2_OFF 82944
#define SMEM_DZN2_STAGE_BYTES 4096
#define SMEM_DZN2_STRIDE 86016
#define SMEM_TOTAL 232448
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
kernel_cake_mhc_train_bwd_slab__xrsg(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap dY, const __grid_constant__ CUtensorMap dU, const __grid_constant__ CUtensorMap dzp, const __grid_constant__ CUtensorMap fnTp, const __grid_constant__ CUtensorMap dX, float* __restrict__ tokp, float* __restrict__ dfn_part, int T, int P, int rpc)
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
    #define stage_full_z_addr (mbar_base + 32)
    #define stage_full_x_addr (mbar_base + 48)
    #define stage_empty_e_addr (mbar_base + 64)
    #define fn_full_addr (mbar_base + 80)
    #define fn_empty_addr (mbar_base + 88)
    #define acc_full_addr (mbar_base + 96)
    #define acc_empty_addr (mbar_base + 112)
    #define dfn_full_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* wtv = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int wtv_addr = smem + 222208;
    __nv_bfloat16* x0_1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int x0_addr_1_0 = smem + 1024;
    __nv_bfloat16* dy0_1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dy0_addr_1_0 = smem + 33792;
    __nv_bfloat16* x0_1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int x0_addr_1_1 = smem + 9216;
    __nv_bfloat16* dy0_1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int dy0_addr_1_1 = smem + 41984;
    __nv_bfloat16* x0_1_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int x0_addr_1_2 = smem + 17408;
    __nv_bfloat16* dy0_1_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int dy0_addr_1_2 = smem + 50176;
    __nv_bfloat16* x0_1_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int x0_addr_1_3 = smem + 25600;
    __nv_bfloat16* dy0_1_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int dy0_addr_1_3 = smem + 58368;
    __nv_bfloat16* dus = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int dus_addr = smem + 66560;
    __nv_bfloat16* dz0_1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int dz0_addr_1_0 = smem + 74752;
    __nv_bfloat16* dz0_1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 78848);
    const int dz0_addr_1_1 = smem + 78848;
    __nv_bfloat16* dz0_1_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int dz0_addr_1_2 = smem + 82944;
    __nv_bfloat16* fn0_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int fn0_0_addr_1_0 = smem + 173056;
    __nv_bfloat16* fn0_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 181248);
    const int fn0_0_addr_1_1 = smem + 181248;
    __nv_bfloat16* fn1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int fn1_0_addr_1_0 = smem + 189440;
    __nv_bfloat16* fn1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int fn1_0_addr_1_1 = smem + 197632;
    __nv_bfloat16* fn2_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int fn2_0_addr_1_0 = smem + 205824;
    __nv_bfloat16* fn2_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int fn2_0_addr_1_1 = smem + 214016;
    __nv_bfloat16* fnw0_1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int fnw0_addr_1_0 = smem + 173056;
    __nv_bfloat16* fnw0_1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 189440);
    const int fnw0_addr_1_1 = smem + 189440;
    __nv_bfloat16* fnw0_1_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int fnw0_addr_1_2 = smem + 205824;
    __nv_bfloat16* xa0_1_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int xa0_addr_1_0 = smem + 1024;
    __nv_bfloat16* xa0_1_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int xa0_addr_1_1 = smem + 17408;
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

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[0..144)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'stage_pipe' ---
            // stage_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // stage_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            mbarrier_init(smem + 24, 8);
            // stage_full_z: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // stage_full_x: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // stage_empty_e: 2 barriers, init_count=8
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            // --- pipeline 'fn_pipe' ---
            // fn_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // fn_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // acc_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            // dfn_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 144);
    if (warp == 0) {
        int _tmem_hold = smem + 144;
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
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
    }

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int _phase_fn_empty = 1;
            unsigned int _phase_stage_empty_e = 1;
            if (elect_sync()) {
                const int bid_0 = (int)bid;
                const int grid = (int)num_bids;
                const int n_mine = (64 * P - bid_0 + grid - 1) / grid;
                unsigned int ld_stage = 0;
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
                        mbarrier_wait(fn_empty_addr, _phase_fn_empty);
                        tma_3d_gmem2smem(fn0_0_addr_1_0, (&fnTp), 0, k0, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_0_addr_1_0 + 4096, (&fnTp), 0, k0 + 4096, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_0_addr_1_1, (&fnTp), 0, k0 + 8192, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_0_addr_1_1 + 4096, (&fnTp), 0, k0 + 12288, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr_1_0, (&fnTp), 0, k0, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr_1_0 + 4096, (&fnTp), 0, k0 + 4096, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr_1_1, (&fnTp), 0, k0 + 8192, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr_1_1 + 4096, (&fnTp), 0, k0 + 12288, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr_1_0, (&fnTp), 0, k0, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr_1_0 + 4096, (&fnTp), 0, k0 + 4096, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr_1_1, (&fnTp), 0, k0 + 8192, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr_1_1 + 4096, (&fnTp), 0, k0 + 12288, 2, fn_full_addr);
                        mbarrier_arrive_expect_tx(fn_full_addr, 49152);
                        _phase_fn_empty ^= 1;
                        #pragma unroll 1
                        for (int tile = 0; tile < n_tiles; tile++) {
                            mbarrier_wait(stage_empty_e_addr + (ld_stage) * 8, _phase_stage_empty_e);
                            int t0 = row0 + tile * 64;
                            tma_2d_gmem2smem(dz0_addr_1_0 + ld_stage * 86016, (&dzp), 0, t0, stage_full_z_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz0_addr_1_1 + ld_stage * 86016, (&dzp), 32, t0, stage_full_z_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz0_addr_1_2 + ld_stage * 86016, (&dzp), 64, t0, stage_full_z_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_z_addr + (ld_stage) * 8, 12288);
                            tma_2d_gmem2smem(dus_addr + ld_stage * 86016, (&dU), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr_1_0 + ld_stage * 86016, (&dY), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr_1_1 + ld_stage * 86016, (&dY), k0 + 4096, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr_1_2 + ld_stage * 86016, (&dY), k0 + 8192, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr_1_3 + ld_stage * 86016, (&dY), k0 + 12288, t0, stage_full_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_addr + (ld_stage) * 8, 40960);
                            mbarrier_wait(stage_empty_addr + (ld_stage) * 8, _phase_stage_empty_e);
                            tma_2d_gmem2smem(x0_addr_1_0 + ld_stage * 86016, (&X), k0, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x0_addr_1_1 + ld_stage * 86016, (&X), k0 + 4096, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x0_addr_1_2 + ld_stage * 86016, (&X), k0 + 8192, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x0_addr_1_3 + ld_stage * 86016, (&X), k0 + 12288, t0, stage_full_x_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_x_addr + (ld_stage) * 8, 32768);
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
                                    int s_0 = u_0 / P;
                                    int p_0 = u_0 - s_0 * P;
                                    int row0_0 = p_0 * rpc;
                                    int rows_0 = T - row0_0;
                                    if (rows_0 > rpc) {
                                        rows_0 = rpc;
                                    }
                                    int n_tiles_0 = 0;
                                    if (rows_0 > 0) {
                                        n_tiles_0 = (rows_0 + 63) / 64;
                                    }
                                    int tt = nt - n_tiles;
                                    if (tt < n_tiles_0) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 4096)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 4096)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 8192)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 8192)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 12288)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 12288)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dU))), "r"((int)(s_0 * 64)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(0)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(32)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(64)), "r"((int)(row0_0 + tt * 64)) : "memory");
                                    }
                                }
                            }
                            ld_stage += 1;
                            if (ld_stage == 2) { ld_stage = 0; _phase_stage_empty_e ^= 1; }
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
            unsigned int grp_mm = 0;
            int tcm = 0;
            unsigned int _phase_fn_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_stage_full_z = 0;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < n_mine_1; j_1++) {
                int u_1 = bid_0_1 + j_1 * grid_1;
                int s_1 = u_1 / P;
                int p_1 = u_1 - s_1 * P;
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
                    mbarrier_wait(fn_full_addr, _phase_fn_full);
                    #pragma unroll 1
                    for (int tile_1 = 0; tile_1 < n_tiles_1; tile_1++) {
                        mbarrier_wait(acc_empty_addr + (acc_mm) * 8, _phase_acc_empty);
                        mbarrier_wait(stage_full_z_addr + (mm_stage) * 8, _phase_stage_full_z);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_0 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_0 = make_warp_uniform(((fnw0_addr_1_2) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_0 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_0 = make_warp_uniform(((fnw0_addr_1_2) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accw + (acc_mm * 128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = make_warp_uniform((((dz0_addr_1_2) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_1 = make_warp_uniform(((fnw0_addr_1_0) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_1 = make_warp_uniform((((dz0_addr_1_2) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_1 = make_warp_uniform(((fnw0_addr_1_0) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_2 = make_warp_uniform((((dz0_addr_1_1) >> 4) & 0x7FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_2 = make_warp_uniform(((fnw0_addr_1_1) >> 4) & 0x7FFF);
#else
                        int _mma_a_lo_2 = make_warp_uniform((((dz0_addr_1_1) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_2 = make_warp_uniform(((fnw0_addr_1_1) >> 4) & 0x3FFF);
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_accw + (acc_mm * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_3 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_3 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_2), "r"((tmem_accw + (acc_mm * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_4 = make_warp_uniform((((dz0_addr_1_1) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_4 = make_warp_uniform((((dz0_addr_1_1) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_5 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x7FFF) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_5 = make_warp_uniform((((dz0_addr_1_0) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_1), "r"((tmem_accw + (acc_mm * 128))), "r"(1) : "memory");
                        elect_commit(acc_full_addr + (acc_mm) * 8);
                        mbarrier_wait(stage_full_x_addr + (mm_stage) * 8, _phase_stage_full_z);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int in_grp = tile_1 % 16;
                        int small_init = ((in_grp == 0) ? 1 : 0);
                        unsigned int sbuf = grp_mm % 2;
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_6 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_3 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_6 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_3 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_3), "r"((tmem_dfb0 + (acc_mm * 32))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_7 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_4 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_7 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_4 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_4), "r"((tmem_dfs0 + (sbuf * 32))), "r"(((small_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_8 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_5 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_8 = make_warp_uniform(((((xa0_addr_1_0) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_5 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_5), "r"((tmem_dfs0 + (sbuf * 32))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_9 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_6 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_9 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_6), "r"((tmem_dfb1 + (acc_mm * 32))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_10 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_7 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_10 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_7), "r"((tmem_dfs1 + (sbuf * 32))), "r"(((small_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_11 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x7FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_8 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x7FFF) | 0x1000000) + (mm_stage) * 5376);
#else
                        int _mma_a_lo_11 = make_warp_uniform(((((xa0_addr_1_1) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_8), "r"((tmem_dfs1 + (sbuf * 32))), "r"(1) : "memory");
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
                        elect_commit(dfn_full_addr + (acc_mm) * 8);
                        acc_mm += 1;
                        if (acc_mm == 2) { acc_mm = 0; _phase_acc_empty ^= 1; }
                        mm_stage += 1;
                        if (mm_stage == 2) { mm_stage = 0; _phase_stage_full_z ^= 1; }
                    }
                    elect_commit(fn_empty_addr);
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
            const int tokbase = q % 2 * 32 + mt * 16;
            const unsigned int wtab = wtv_addr + (unsigned int)(ew * 1280);
            const unsigned int wpre_off = (unsigned int)(8 + 4 * i0);
            const unsigned int wc_off = (unsigned int)(24 + 8 * i0);
            const unsigned int wrow_bytes = (unsigned int)(tokbase * 128);
            const int wcoord_t = tokbase;
            int wq_r4[2];
            int wq_q4[2];
            #pragma unroll
            for (int kq = 0; kq < 2; kq++) {
                int cq = lane_1 + kq * 32;
                wq_r4[kq] = cq / 4;
                wq_q4[kq] = cq - cq / 4 * 4;
            }
            int wq_r5[3];
            int wq_s5[3];
            int wq_q5[3];
            #pragma unroll
            for (int kq_1 = 0; kq_1 < 3; kq_1++) {
                int cq_1 = lane_1 + kq_1 * 32;
                wq_r5[kq_1] = cq_1 / 5;
                wq_s5[kq_1] = cq_1 - cq_1 / 5 * 5;
                wq_q5[kq_1] = wq_s5[kq_1] + (wq_s5[kq_1] + 2) / 4;
            }
            int qa_pre = 0;
            unsigned int _phase_acc_full = 0;
            unsigned int _phase_stage_full_x = 0;
            #pragma unroll 1
            for (int j_2 = 0; j_2 < n_mine_2; j_2++) {
                int u_2 = bid_0_2 + j_2 * grid_2;
                int s_2 = u_2 / P;
                int p_2 = u_2 - s_2 * P;
                int row0_2 = p_2 * rpc;
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
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < n_tiles_2; tile_2++) {
                    int t0_1 = row0_2 + tile_2 * 64;
                    if (tile_2 == 0) {
                        if (qa_pre == 0) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            __syncwarp();
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (i0 == 0) {
                                #pragma unroll
                                for (int kq_2 = 0; kq_2 < 2; kq_2++) {
                                    int rowa = t0_1 + tokbase + wq_r4[kq_2];
                                    if (rowa >= T) {
                                        rowa = T - 1;
                                    }
                                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                        :: "r"(wtab + (unsigned int)(wq_r4[kq_2] * 80 + wq_q4[kq_2] * 16)), "l"(tokp + ((unsigned long long)rowa * 24 + (unsigned long long)(wq_q4[kq_2] * 4))));
                                }
                            } else {
                                #pragma unroll
                                for (int kq_3 = 0; kq_3 < 3; kq_3++) {
                                    int rowa_1 = t0_1 + tokbase + wq_r5[kq_3];
                                    if (rowa_1 >= T) {
                                        rowa_1 = T - 1;
                                    }
                                    if (kq_3 == 2) {
                                        asm volatile(
                                            "{\n\t"
                                            ".reg .pred p;\n\t"
                                            "setp.ne.b32 p, %0, 0;\n\t"
                                            "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                            "}"
                                            :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_3] * 80 + wq_s5[kq_3] * 16)), "l"(tokp + ((unsigned long long)rowa_1 * 24 + (unsigned long long)(wq_q5[kq_3] * 4))));
                                    } else {
                                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                            :: "r"(wtab + (unsigned int)(wq_r5[kq_3] * 80 + wq_s5[kq_3] * 16)), "l"(tokp + ((unsigned long long)rowa_1 * 24 + (unsigned long long)(wq_q5[kq_3] * 4))));
                                    }
                                }
                            }
                            asm volatile("cp.async.commit_group;");
                        }
                        qa_pre = 0;
                    }
                    mbarrier_wait(acc_full_addr + (acc_ep) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int tm = lane_tm + acc_ep * 64;
                    unsigned int tab = dz0_addr_1_0 + ep_stage * 86016;
                    unsigned int tmc = acc_ep * 128;
                    float tmem_load[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load[15]))
                        : "r"(tmw + tmc));
                    float tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1[15]))
                        : "r"(tmw + tmc + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("cp.async.wait_group 0;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sbase = x0_addr_1_0 + ep_stage * 86016;
                    unsigned int tv[24];
                    unsigned int wpp[2];
                    unsigned int wrow = wtab + (unsigned int)(tq * 80);
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[0])) : "r"(wrow + 4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[6])) : "r"(wrow + 4));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wpp[0])), "=r"(*reinterpret_cast<uint32_t*>(&wpp[(0) + 1]))
                        : "r"(wrow + wpre_off));
                    tv[1] = wpp[0];
                    tv[7] = wpp[1];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(2) + 1]))
                        : "r"(wrow + wc_off));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(4) + 1]))
                        : "r"(wrow + wc_off + 8));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(8) + 1]))
                        : "r"(wrow + wc_off + 16));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[10])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(10) + 1]))
                        : "r"(wrow + wc_off + 24));
                    unsigned int wrow_0 = wtab + (unsigned int)((tq + 8) * 80);
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[12])) : "r"(wrow_0 + 4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tv[18])) : "r"(wrow_0 + 4));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wpp[0])), "=r"(*reinterpret_cast<uint32_t*>(&wpp[(0) + 1]))
                        : "r"(wrow_0 + wpre_off));
                    tv[13] = wpp[0];
                    tv[19] = wpp[1];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[14])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(14) + 1]))
                        : "r"(wrow_0 + wc_off));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[16])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(16) + 1]))
                        : "r"(wrow_0 + wc_off + 8));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[20])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(20) + 1]))
                        : "r"(wrow_0 + wc_off + 16));
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tv[22])), "=r"(*reinterpret_cast<uint32_t*>(&tv[(22) + 1]))
                        : "r"(wrow_0 + wc_off + 24));
                    float nct[2];
                    nct[0] = -__uint_as_float(tv[0]);
                    nct[1] = -__uint_as_float(tv[12]);
                    unsigned int wv0[5];
                    unsigned int wv1[5];
                    unsigned int xr0[2];
                    unsigned int xr1[2];
                    mbarrier_wait(stage_full_x_addr + (ep_stage) * 8, _phase_stage_full_x);
                    mbarrier_wait(stage_full_addr + (ep_stage) * 8, _phase_stage_full_x);
                    mbarrier_wait(dfn_full_addr + (acc_ep) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int rb0_1_0 = sbase + offg[0] + rowb0;
                    unsigned int rb1_1_0 = rb0_1_0 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_0 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_0 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_0 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_0 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_0 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_0 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_0 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_0 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_0 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_0 + 57344));
                    unsigned int xw0_1_0 = rb0_1_0 + xoff0;
                    unsigned int xw1_1_0 = rb1_1_0 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_0));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_0));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_0 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_0 + 8192));
                    float dul_1_0 = __uint_as_float(wv0[0] << 16);
                    float duh_1_0 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_0[4];
                    float dyh_1_0[4];
                    dyl_1_0[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_0[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_0[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_0[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_0[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_0[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_0[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_0[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_0 = __uint_as_float(tv[1]);
                    float c0_1_0 = __uint_as_float(tv[2]);
                    float c1_1_0 = __uint_as_float(tv[3]);
                    float c2_1_0 = __uint_as_float(tv[4]);
                    float c3_1_0 = __uint_as_float(tv[5]);
                    float rl0_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_0) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load[0]));
                    float rh0_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_0) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load[1]));
                    float rl1_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_0) : "f"(pre_i_2_0), "f"(dul_1_0), "f"(rl0_1_0));
                    float rh1_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_0) : "f"(pre_i_2_0), "f"(duh_1_0), "f"(rh0_1_0));
                    float rl2_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_0) : "f"(c0_1_0), "f"(dyl_1_0[0]), "f"(rl1_1_0));
                    float rh2_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_0) : "f"(c0_1_0), "f"(dyh_1_0[0]), "f"(rh1_1_0));
                    float rl3_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_0) : "f"(c1_1_0), "f"(dyl_1_0[1]), "f"(rl2_1_0));
                    float rh3_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_0) : "f"(c1_1_0), "f"(dyh_1_0[1]), "f"(rh2_1_0));
                    float rl4_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_0) : "f"(c2_1_0), "f"(dyl_1_0[2]), "f"(rl3_1_0));
                    float rh4_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_0) : "f"(c2_1_0), "f"(dyh_1_0[2]), "f"(rh3_1_0));
                    float rl5_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_0) : "f"(c3_1_0), "f"(dyl_1_0[3]), "f"(rl4_1_0));
                    float rh5_1_0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_0) : "f"(c3_1_0), "f"(dyh_1_0[3]), "f"(rh4_1_0));
                    unsigned int word_1_0 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_0, rh5_1_0)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_0), "r"(word_1_0) : "memory");
                    float pre_i_2_1 = __uint_as_float(tv[7]);
                    float c0_1_1 = __uint_as_float(tv[8]);
                    float c1_1_1 = __uint_as_float(tv[9]);
                    float c2_1_1 = __uint_as_float(tv[10]);
                    float c3_1_1 = __uint_as_float(tv[11]);
                    float rl0_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_1) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1[0]));
                    float rh0_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_1) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1[1]));
                    float rl1_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_1) : "f"(pre_i_2_1), "f"(dul_1_0), "f"(rl0_1_1));
                    float rh1_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_1) : "f"(pre_i_2_1), "f"(duh_1_0), "f"(rh0_1_1));
                    float rl2_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_1) : "f"(c0_1_1), "f"(dyl_1_0[0]), "f"(rl1_1_1));
                    float rh2_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_1) : "f"(c0_1_1), "f"(dyh_1_0[0]), "f"(rh1_1_1));
                    float rl3_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_1) : "f"(c1_1_1), "f"(dyl_1_0[1]), "f"(rl2_1_1));
                    float rh3_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_1) : "f"(c1_1_1), "f"(dyh_1_0[1]), "f"(rh2_1_1));
                    float rl4_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_1) : "f"(c2_1_1), "f"(dyl_1_0[2]), "f"(rl3_1_1));
                    float rh4_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_1) : "f"(c2_1_1), "f"(dyh_1_0[2]), "f"(rh3_1_1));
                    float rl5_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_1) : "f"(c3_1_1), "f"(dyl_1_0[3]), "f"(rl4_1_1));
                    float rh5_1_1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_1) : "f"(c3_1_1), "f"(dyh_1_0[3]), "f"(rh4_1_1));
                    unsigned int word_1_1 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_1, rh5_1_1)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_0 + 8192), "r"(word_1_1) : "memory");
                    float dul_1_1 = __uint_as_float(wv1[0] << 16);
                    float duh_1_1 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_1[4];
                    float dyh_1_1[4];
                    dyl_1_1[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_1[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_1[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_1[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_1[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_1[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_1[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_1[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_2 = __uint_as_float(tv[13]);
                    float c0_1_2 = __uint_as_float(tv[14]);
                    float c1_1_2 = __uint_as_float(tv[15]);
                    float c2_1_2 = __uint_as_float(tv[16]);
                    float c3_1_2 = __uint_as_float(tv[17]);
                    float rl0_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_2) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load[2]));
                    float rh0_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_2) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load[3]));
                    float rl1_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_2) : "f"(pre_i_2_2), "f"(dul_1_1), "f"(rl0_1_2));
                    float rh1_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_2) : "f"(pre_i_2_2), "f"(duh_1_1), "f"(rh0_1_2));
                    float rl2_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_2) : "f"(c0_1_2), "f"(dyl_1_1[0]), "f"(rl1_1_2));
                    float rh2_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_2) : "f"(c0_1_2), "f"(dyh_1_1[0]), "f"(rh1_1_2));
                    float rl3_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_2) : "f"(c1_1_2), "f"(dyl_1_1[1]), "f"(rl2_1_2));
                    float rh3_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_2) : "f"(c1_1_2), "f"(dyh_1_1[1]), "f"(rh2_1_2));
                    float rl4_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_2) : "f"(c2_1_2), "f"(dyl_1_1[2]), "f"(rl3_1_2));
                    float rh4_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_2) : "f"(c2_1_2), "f"(dyh_1_1[2]), "f"(rh3_1_2));
                    float rl5_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_2) : "f"(c3_1_2), "f"(dyl_1_1[3]), "f"(rl4_1_2));
                    float rh5_1_2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_2) : "f"(c3_1_2), "f"(dyh_1_1[3]), "f"(rh4_1_2));
                    unsigned int word_1_2 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_2, rh5_1_2)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_0), "r"(word_1_2) : "memory");
                    float pre_i_2_3 = __uint_as_float(tv[19]);
                    float c0_1_3 = __uint_as_float(tv[20]);
                    float c1_1_3 = __uint_as_float(tv[21]);
                    float c2_1_3 = __uint_as_float(tv[22]);
                    float c3_1_3 = __uint_as_float(tv[23]);
                    float rl0_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_3) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1[2]));
                    float rh0_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_3) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1[3]));
                    float rl1_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_3) : "f"(pre_i_2_3), "f"(dul_1_1), "f"(rl0_1_3));
                    float rh1_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_3) : "f"(pre_i_2_3), "f"(duh_1_1), "f"(rh0_1_3));
                    float rl2_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_3) : "f"(c0_1_3), "f"(dyl_1_1[0]), "f"(rl1_1_3));
                    float rh2_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_3) : "f"(c0_1_3), "f"(dyh_1_1[0]), "f"(rh1_1_3));
                    float rl3_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_3) : "f"(c1_1_3), "f"(dyl_1_1[1]), "f"(rl2_1_3));
                    float rh3_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_3) : "f"(c1_1_3), "f"(dyh_1_1[1]), "f"(rh2_1_3));
                    float rl4_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_3) : "f"(c2_1_3), "f"(dyl_1_1[2]), "f"(rl3_1_3));
                    float rh4_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_3) : "f"(c2_1_3), "f"(dyh_1_1[2]), "f"(rh3_1_3));
                    float rl5_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_3) : "f"(c3_1_3), "f"(dyl_1_1[3]), "f"(rl4_1_3));
                    float rh5_1_3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_3) : "f"(c3_1_3), "f"(dyh_1_1[3]), "f"(rh4_1_3));
                    unsigned int word_1_3 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_3, rh5_1_3)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_0 + 8192), "r"(word_1_3) : "memory");
                    unsigned int rb0_1_1 = sbase + offg[1] + rowb0;
                    unsigned int rb1_1_1 = rb0_1_1 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_1 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_1 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_1 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_1 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_1 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_1 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_1 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_1 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_1 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_1 + 57344));
                    unsigned int xw0_1_1 = rb0_1_1 + xoff0;
                    unsigned int xw1_1_1 = rb1_1_1 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_1));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_1));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_1 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_1 + 8192));
                    float dul_1_2 = __uint_as_float(wv0[0] << 16);
                    float duh_1_2 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_2[4];
                    float dyh_1_2[4];
                    dyl_1_2[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_2[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_2[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_2[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_2[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_2[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_2[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_2[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_4 = __uint_as_float(tv[1]);
                    float c0_1_4 = __uint_as_float(tv[2]);
                    float c1_1_4 = __uint_as_float(tv[3]);
                    float c2_1_4 = __uint_as_float(tv[4]);
                    float c3_1_4 = __uint_as_float(tv[5]);
                    float rl0_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_4) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load[4]));
                    float rh0_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_4) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load[5]));
                    float rl1_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_4) : "f"(pre_i_2_4), "f"(dul_1_2), "f"(rl0_1_4));
                    float rh1_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_4) : "f"(pre_i_2_4), "f"(duh_1_2), "f"(rh0_1_4));
                    float rl2_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_4) : "f"(c0_1_4), "f"(dyl_1_2[0]), "f"(rl1_1_4));
                    float rh2_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_4) : "f"(c0_1_4), "f"(dyh_1_2[0]), "f"(rh1_1_4));
                    float rl3_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_4) : "f"(c1_1_4), "f"(dyl_1_2[1]), "f"(rl2_1_4));
                    float rh3_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_4) : "f"(c1_1_4), "f"(dyh_1_2[1]), "f"(rh2_1_4));
                    float rl4_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_4) : "f"(c2_1_4), "f"(dyl_1_2[2]), "f"(rl3_1_4));
                    float rh4_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_4) : "f"(c2_1_4), "f"(dyh_1_2[2]), "f"(rh3_1_4));
                    float rl5_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_4) : "f"(c3_1_4), "f"(dyl_1_2[3]), "f"(rl4_1_4));
                    float rh5_1_4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_4) : "f"(c3_1_4), "f"(dyh_1_2[3]), "f"(rh4_1_4));
                    unsigned int word_1_4 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_4, rh5_1_4)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_1), "r"(word_1_4) : "memory");
                    float pre_i_2_5 = __uint_as_float(tv[7]);
                    float c0_1_5 = __uint_as_float(tv[8]);
                    float c1_1_5 = __uint_as_float(tv[9]);
                    float c2_1_5 = __uint_as_float(tv[10]);
                    float c3_1_5 = __uint_as_float(tv[11]);
                    float rl0_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_5) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1[4]));
                    float rh0_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_5) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1[5]));
                    float rl1_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_5) : "f"(pre_i_2_5), "f"(dul_1_2), "f"(rl0_1_5));
                    float rh1_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_5) : "f"(pre_i_2_5), "f"(duh_1_2), "f"(rh0_1_5));
                    float rl2_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_5) : "f"(c0_1_5), "f"(dyl_1_2[0]), "f"(rl1_1_5));
                    float rh2_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_5) : "f"(c0_1_5), "f"(dyh_1_2[0]), "f"(rh1_1_5));
                    float rl3_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_5) : "f"(c1_1_5), "f"(dyl_1_2[1]), "f"(rl2_1_5));
                    float rh3_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_5) : "f"(c1_1_5), "f"(dyh_1_2[1]), "f"(rh2_1_5));
                    float rl4_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_5) : "f"(c2_1_5), "f"(dyl_1_2[2]), "f"(rl3_1_5));
                    float rh4_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_5) : "f"(c2_1_5), "f"(dyh_1_2[2]), "f"(rh3_1_5));
                    float rl5_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_5) : "f"(c3_1_5), "f"(dyl_1_2[3]), "f"(rl4_1_5));
                    float rh5_1_5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_5) : "f"(c3_1_5), "f"(dyh_1_2[3]), "f"(rh4_1_5));
                    unsigned int word_1_5 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_5, rh5_1_5)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_1 + 8192), "r"(word_1_5) : "memory");
                    float dul_1_3 = __uint_as_float(wv1[0] << 16);
                    float duh_1_3 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_3[4];
                    float dyh_1_3[4];
                    dyl_1_3[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_3[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_3[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_3[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_3[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_3[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_3[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_3[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_6 = __uint_as_float(tv[13]);
                    float c0_1_6 = __uint_as_float(tv[14]);
                    float c1_1_6 = __uint_as_float(tv[15]);
                    float c2_1_6 = __uint_as_float(tv[16]);
                    float c3_1_6 = __uint_as_float(tv[17]);
                    float rl0_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_6) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load[6]));
                    float rh0_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_6) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load[7]));
                    float rl1_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_6) : "f"(pre_i_2_6), "f"(dul_1_3), "f"(rl0_1_6));
                    float rh1_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_6) : "f"(pre_i_2_6), "f"(duh_1_3), "f"(rh0_1_6));
                    float rl2_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_6) : "f"(c0_1_6), "f"(dyl_1_3[0]), "f"(rl1_1_6));
                    float rh2_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_6) : "f"(c0_1_6), "f"(dyh_1_3[0]), "f"(rh1_1_6));
                    float rl3_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_6) : "f"(c1_1_6), "f"(dyl_1_3[1]), "f"(rl2_1_6));
                    float rh3_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_6) : "f"(c1_1_6), "f"(dyh_1_3[1]), "f"(rh2_1_6));
                    float rl4_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_6) : "f"(c2_1_6), "f"(dyl_1_3[2]), "f"(rl3_1_6));
                    float rh4_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_6) : "f"(c2_1_6), "f"(dyh_1_3[2]), "f"(rh3_1_6));
                    float rl5_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_6) : "f"(c3_1_6), "f"(dyl_1_3[3]), "f"(rl4_1_6));
                    float rh5_1_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_6) : "f"(c3_1_6), "f"(dyh_1_3[3]), "f"(rh4_1_6));
                    unsigned int word_1_6 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_6, rh5_1_6)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_1), "r"(word_1_6) : "memory");
                    float pre_i_2_7 = __uint_as_float(tv[19]);
                    float c0_1_7 = __uint_as_float(tv[20]);
                    float c1_1_7 = __uint_as_float(tv[21]);
                    float c2_1_7 = __uint_as_float(tv[22]);
                    float c3_1_7 = __uint_as_float(tv[23]);
                    float rl0_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_7) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1[6]));
                    float rh0_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_7) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1[7]));
                    float rl1_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_7) : "f"(pre_i_2_7), "f"(dul_1_3), "f"(rl0_1_7));
                    float rh1_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_7) : "f"(pre_i_2_7), "f"(duh_1_3), "f"(rh0_1_7));
                    float rl2_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_7) : "f"(c0_1_7), "f"(dyl_1_3[0]), "f"(rl1_1_7));
                    float rh2_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_7) : "f"(c0_1_7), "f"(dyh_1_3[0]), "f"(rh1_1_7));
                    float rl3_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_7) : "f"(c1_1_7), "f"(dyl_1_3[1]), "f"(rl2_1_7));
                    float rh3_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_7) : "f"(c1_1_7), "f"(dyh_1_3[1]), "f"(rh2_1_7));
                    float rl4_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_7) : "f"(c2_1_7), "f"(dyl_1_3[2]), "f"(rl3_1_7));
                    float rh4_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_7) : "f"(c2_1_7), "f"(dyh_1_3[2]), "f"(rh3_1_7));
                    float rl5_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_7) : "f"(c3_1_7), "f"(dyl_1_3[3]), "f"(rl4_1_7));
                    float rh5_1_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_7) : "f"(c3_1_7), "f"(dyh_1_3[3]), "f"(rh4_1_7));
                    unsigned int word_1_7 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_7, rh5_1_7)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_1 + 8192), "r"(word_1_7) : "memory");
                    unsigned int rb0_1_2 = sbase + offg[2] + rowb0;
                    unsigned int rb1_1_2 = rb0_1_2 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_2 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_2 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_2 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_2 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_2 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_2 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_2 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_2 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_2 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_2 + 57344));
                    unsigned int xw0_1_2 = rb0_1_2 + xoff0;
                    unsigned int xw1_1_2 = rb1_1_2 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_2));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_2));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_2 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_2 + 8192));
                    float dul_1_4 = __uint_as_float(wv0[0] << 16);
                    float duh_1_4 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_4[4];
                    float dyh_1_4[4];
                    dyl_1_4[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_4[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_4[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_4[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_4[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_4[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_4[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_4[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_8 = __uint_as_float(tv[1]);
                    float c0_1_8 = __uint_as_float(tv[2]);
                    float c1_1_8 = __uint_as_float(tv[3]);
                    float c2_1_8 = __uint_as_float(tv[4]);
                    float c3_1_8 = __uint_as_float(tv[5]);
                    float rl0_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_8) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load[8]));
                    float rh0_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_8) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load[9]));
                    float rl1_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_8) : "f"(pre_i_2_8), "f"(dul_1_4), "f"(rl0_1_8));
                    float rh1_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_8) : "f"(pre_i_2_8), "f"(duh_1_4), "f"(rh0_1_8));
                    float rl2_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_8) : "f"(c0_1_8), "f"(dyl_1_4[0]), "f"(rl1_1_8));
                    float rh2_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_8) : "f"(c0_1_8), "f"(dyh_1_4[0]), "f"(rh1_1_8));
                    float rl3_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_8) : "f"(c1_1_8), "f"(dyl_1_4[1]), "f"(rl2_1_8));
                    float rh3_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_8) : "f"(c1_1_8), "f"(dyh_1_4[1]), "f"(rh2_1_8));
                    float rl4_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_8) : "f"(c2_1_8), "f"(dyl_1_4[2]), "f"(rl3_1_8));
                    float rh4_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_8) : "f"(c2_1_8), "f"(dyh_1_4[2]), "f"(rh3_1_8));
                    float rl5_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_8) : "f"(c3_1_8), "f"(dyl_1_4[3]), "f"(rl4_1_8));
                    float rh5_1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_8) : "f"(c3_1_8), "f"(dyh_1_4[3]), "f"(rh4_1_8));
                    unsigned int word_1_8 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_8, rh5_1_8)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_2), "r"(word_1_8) : "memory");
                    float pre_i_2_9 = __uint_as_float(tv[7]);
                    float c0_1_9 = __uint_as_float(tv[8]);
                    float c1_1_9 = __uint_as_float(tv[9]);
                    float c2_1_9 = __uint_as_float(tv[10]);
                    float c3_1_9 = __uint_as_float(tv[11]);
                    float rl0_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_9) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1[8]));
                    float rh0_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_9) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1[9]));
                    float rl1_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_9) : "f"(pre_i_2_9), "f"(dul_1_4), "f"(rl0_1_9));
                    float rh1_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_9) : "f"(pre_i_2_9), "f"(duh_1_4), "f"(rh0_1_9));
                    float rl2_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_9) : "f"(c0_1_9), "f"(dyl_1_4[0]), "f"(rl1_1_9));
                    float rh2_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_9) : "f"(c0_1_9), "f"(dyh_1_4[0]), "f"(rh1_1_9));
                    float rl3_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_9) : "f"(c1_1_9), "f"(dyl_1_4[1]), "f"(rl2_1_9));
                    float rh3_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_9) : "f"(c1_1_9), "f"(dyh_1_4[1]), "f"(rh2_1_9));
                    float rl4_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_9) : "f"(c2_1_9), "f"(dyl_1_4[2]), "f"(rl3_1_9));
                    float rh4_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_9) : "f"(c2_1_9), "f"(dyh_1_4[2]), "f"(rh3_1_9));
                    float rl5_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_9) : "f"(c3_1_9), "f"(dyl_1_4[3]), "f"(rl4_1_9));
                    float rh5_1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_9) : "f"(c3_1_9), "f"(dyh_1_4[3]), "f"(rh4_1_9));
                    unsigned int word_1_9 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_9, rh5_1_9)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_2 + 8192), "r"(word_1_9) : "memory");
                    float dul_1_5 = __uint_as_float(wv1[0] << 16);
                    float duh_1_5 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_5[4];
                    float dyh_1_5[4];
                    dyl_1_5[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_5[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_5[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_5[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_5[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_5[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_5[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_5[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_10 = __uint_as_float(tv[13]);
                    float c0_1_10 = __uint_as_float(tv[14]);
                    float c1_1_10 = __uint_as_float(tv[15]);
                    float c2_1_10 = __uint_as_float(tv[16]);
                    float c3_1_10 = __uint_as_float(tv[17]);
                    float rl0_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_10) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load[10]));
                    float rh0_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_10) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load[11]));
                    float rl1_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_10) : "f"(pre_i_2_10), "f"(dul_1_5), "f"(rl0_1_10));
                    float rh1_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_10) : "f"(pre_i_2_10), "f"(duh_1_5), "f"(rh0_1_10));
                    float rl2_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_10) : "f"(c0_1_10), "f"(dyl_1_5[0]), "f"(rl1_1_10));
                    float rh2_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_10) : "f"(c0_1_10), "f"(dyh_1_5[0]), "f"(rh1_1_10));
                    float rl3_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_10) : "f"(c1_1_10), "f"(dyl_1_5[1]), "f"(rl2_1_10));
                    float rh3_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_10) : "f"(c1_1_10), "f"(dyh_1_5[1]), "f"(rh2_1_10));
                    float rl4_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_10) : "f"(c2_1_10), "f"(dyl_1_5[2]), "f"(rl3_1_10));
                    float rh4_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_10) : "f"(c2_1_10), "f"(dyh_1_5[2]), "f"(rh3_1_10));
                    float rl5_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_10) : "f"(c3_1_10), "f"(dyl_1_5[3]), "f"(rl4_1_10));
                    float rh5_1_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_10) : "f"(c3_1_10), "f"(dyh_1_5[3]), "f"(rh4_1_10));
                    unsigned int word_1_10 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_10, rh5_1_10)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_2), "r"(word_1_10) : "memory");
                    float pre_i_2_11 = __uint_as_float(tv[19]);
                    float c0_1_11 = __uint_as_float(tv[20]);
                    float c1_1_11 = __uint_as_float(tv[21]);
                    float c2_1_11 = __uint_as_float(tv[22]);
                    float c3_1_11 = __uint_as_float(tv[23]);
                    float rl0_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_11) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1[10]));
                    float rh0_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_11) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1[11]));
                    float rl1_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_11) : "f"(pre_i_2_11), "f"(dul_1_5), "f"(rl0_1_11));
                    float rh1_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_11) : "f"(pre_i_2_11), "f"(duh_1_5), "f"(rh0_1_11));
                    float rl2_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_11) : "f"(c0_1_11), "f"(dyl_1_5[0]), "f"(rl1_1_11));
                    float rh2_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_11) : "f"(c0_1_11), "f"(dyh_1_5[0]), "f"(rh1_1_11));
                    float rl3_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_11) : "f"(c1_1_11), "f"(dyl_1_5[1]), "f"(rl2_1_11));
                    float rh3_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_11) : "f"(c1_1_11), "f"(dyh_1_5[1]), "f"(rh2_1_11));
                    float rl4_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_11) : "f"(c2_1_11), "f"(dyl_1_5[2]), "f"(rl3_1_11));
                    float rh4_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_11) : "f"(c2_1_11), "f"(dyh_1_5[2]), "f"(rh3_1_11));
                    float rl5_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_11) : "f"(c3_1_11), "f"(dyl_1_5[3]), "f"(rl4_1_11));
                    float rh5_1_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_11) : "f"(c3_1_11), "f"(dyh_1_5[3]), "f"(rh4_1_11));
                    unsigned int word_1_11 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_11, rh5_1_11)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_2 + 8192), "r"(word_1_11) : "memory");
                    unsigned int rb0_1_3 = sbase + offg[3] + rowb0;
                    unsigned int rb1_1_3 = rb0_1_3 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_3 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_3 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_3 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_3 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_3 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_3 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_3 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_3 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_3 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_3 + 57344));
                    unsigned int xw0_1_3 = rb0_1_3 + xoff0;
                    unsigned int xw1_1_3 = rb1_1_3 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_3));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_3));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_3 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_3 + 8192));
                    float dul_1_6 = __uint_as_float(wv0[0] << 16);
                    float duh_1_6 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_6[4];
                    float dyh_1_6[4];
                    dyl_1_6[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_6[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_6[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_6[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_6[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_6[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_6[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_6[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_12 = __uint_as_float(tv[1]);
                    float c0_1_12 = __uint_as_float(tv[2]);
                    float c1_1_12 = __uint_as_float(tv[3]);
                    float c2_1_12 = __uint_as_float(tv[4]);
                    float c3_1_12 = __uint_as_float(tv[5]);
                    float rl0_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_12) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load[12]));
                    float rh0_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_12) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load[13]));
                    float rl1_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_12) : "f"(pre_i_2_12), "f"(dul_1_6), "f"(rl0_1_12));
                    float rh1_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_12) : "f"(pre_i_2_12), "f"(duh_1_6), "f"(rh0_1_12));
                    float rl2_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_12) : "f"(c0_1_12), "f"(dyl_1_6[0]), "f"(rl1_1_12));
                    float rh2_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_12) : "f"(c0_1_12), "f"(dyh_1_6[0]), "f"(rh1_1_12));
                    float rl3_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_12) : "f"(c1_1_12), "f"(dyl_1_6[1]), "f"(rl2_1_12));
                    float rh3_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_12) : "f"(c1_1_12), "f"(dyh_1_6[1]), "f"(rh2_1_12));
                    float rl4_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_12) : "f"(c2_1_12), "f"(dyl_1_6[2]), "f"(rl3_1_12));
                    float rh4_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_12) : "f"(c2_1_12), "f"(dyh_1_6[2]), "f"(rh3_1_12));
                    float rl5_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_12) : "f"(c3_1_12), "f"(dyl_1_6[3]), "f"(rl4_1_12));
                    float rh5_1_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_12) : "f"(c3_1_12), "f"(dyh_1_6[3]), "f"(rh4_1_12));
                    unsigned int word_1_12 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_12, rh5_1_12)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_3), "r"(word_1_12) : "memory");
                    float pre_i_2_13 = __uint_as_float(tv[7]);
                    float c0_1_13 = __uint_as_float(tv[8]);
                    float c1_1_13 = __uint_as_float(tv[9]);
                    float c2_1_13 = __uint_as_float(tv[10]);
                    float c3_1_13 = __uint_as_float(tv[11]);
                    float rl0_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_13) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1[12]));
                    float rh0_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_13) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1[13]));
                    float rl1_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_13) : "f"(pre_i_2_13), "f"(dul_1_6), "f"(rl0_1_13));
                    float rh1_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_13) : "f"(pre_i_2_13), "f"(duh_1_6), "f"(rh0_1_13));
                    float rl2_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_13) : "f"(c0_1_13), "f"(dyl_1_6[0]), "f"(rl1_1_13));
                    float rh2_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_13) : "f"(c0_1_13), "f"(dyh_1_6[0]), "f"(rh1_1_13));
                    float rl3_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_13) : "f"(c1_1_13), "f"(dyl_1_6[1]), "f"(rl2_1_13));
                    float rh3_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_13) : "f"(c1_1_13), "f"(dyh_1_6[1]), "f"(rh2_1_13));
                    float rl4_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_13) : "f"(c2_1_13), "f"(dyl_1_6[2]), "f"(rl3_1_13));
                    float rh4_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_13) : "f"(c2_1_13), "f"(dyh_1_6[2]), "f"(rh3_1_13));
                    float rl5_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_13) : "f"(c3_1_13), "f"(dyl_1_6[3]), "f"(rl4_1_13));
                    float rh5_1_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_13) : "f"(c3_1_13), "f"(dyh_1_6[3]), "f"(rh4_1_13));
                    unsigned int word_1_13 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_13, rh5_1_13)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_3 + 8192), "r"(word_1_13) : "memory");
                    float dul_1_7 = __uint_as_float(wv1[0] << 16);
                    float duh_1_7 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_7[4];
                    float dyh_1_7[4];
                    dyl_1_7[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_7[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_7[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_7[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_7[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_7[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_7[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_7[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_14 = __uint_as_float(tv[13]);
                    float c0_1_14 = __uint_as_float(tv[14]);
                    float c1_1_14 = __uint_as_float(tv[15]);
                    float c2_1_14 = __uint_as_float(tv[16]);
                    float c3_1_14 = __uint_as_float(tv[17]);
                    float rl0_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_14) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load[14]));
                    float rh0_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_14) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load[15]));
                    float rl1_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_14) : "f"(pre_i_2_14), "f"(dul_1_7), "f"(rl0_1_14));
                    float rh1_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_14) : "f"(pre_i_2_14), "f"(duh_1_7), "f"(rh0_1_14));
                    float rl2_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_14) : "f"(c0_1_14), "f"(dyl_1_7[0]), "f"(rl1_1_14));
                    float rh2_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_14) : "f"(c0_1_14), "f"(dyh_1_7[0]), "f"(rh1_1_14));
                    float rl3_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_14) : "f"(c1_1_14), "f"(dyl_1_7[1]), "f"(rl2_1_14));
                    float rh3_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_14) : "f"(c1_1_14), "f"(dyh_1_7[1]), "f"(rh2_1_14));
                    float rl4_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_14) : "f"(c2_1_14), "f"(dyl_1_7[2]), "f"(rl3_1_14));
                    float rh4_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_14) : "f"(c2_1_14), "f"(dyh_1_7[2]), "f"(rh3_1_14));
                    float rl5_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_14) : "f"(c3_1_14), "f"(dyl_1_7[3]), "f"(rl4_1_14));
                    float rh5_1_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_14) : "f"(c3_1_14), "f"(dyh_1_7[3]), "f"(rh4_1_14));
                    unsigned int word_1_14 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_14, rh5_1_14)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_3), "r"(word_1_14) : "memory");
                    float pre_i_2_15 = __uint_as_float(tv[19]);
                    float c0_1_15 = __uint_as_float(tv[20]);
                    float c1_1_15 = __uint_as_float(tv[21]);
                    float c2_1_15 = __uint_as_float(tv[22]);
                    float c3_1_15 = __uint_as_float(tv[23]);
                    float rl0_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_15) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1[14]));
                    float rh0_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_15) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1[15]));
                    float rl1_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_15) : "f"(pre_i_2_15), "f"(dul_1_7), "f"(rl0_1_15));
                    float rh1_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_15) : "f"(pre_i_2_15), "f"(duh_1_7), "f"(rh0_1_15));
                    float rl2_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_15) : "f"(c0_1_15), "f"(dyl_1_7[0]), "f"(rl1_1_15));
                    float rh2_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_15) : "f"(c0_1_15), "f"(dyh_1_7[0]), "f"(rh1_1_15));
                    float rl3_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_15) : "f"(c1_1_15), "f"(dyl_1_7[1]), "f"(rl2_1_15));
                    float rh3_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_15) : "f"(c1_1_15), "f"(dyh_1_7[1]), "f"(rh2_1_15));
                    float rl4_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_15) : "f"(c2_1_15), "f"(dyl_1_7[2]), "f"(rl3_1_15));
                    float rh4_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_15) : "f"(c2_1_15), "f"(dyh_1_7[2]), "f"(rh3_1_15));
                    float rl5_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_15) : "f"(c3_1_15), "f"(dyl_1_7[3]), "f"(rl4_1_15));
                    float rh5_1_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_15) : "f"(c3_1_15), "f"(dyh_1_7[3]), "f"(rh4_1_15));
                    unsigned int word_1_15 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_15, rh5_1_15)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_3 + 8192), "r"(word_1_15) : "memory");
                    float tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_0[15]))
                        : "r"(tmw + tmc + 32));
                    float tmem_load_1_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&tmem_load_1_0[15]))
                        : "r"(tmw + tmc + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    unsigned int rb0_1_4 = sbase + offg[4] + rowb0;
                    unsigned int rb1_1_4 = rb0_1_4 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_4 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_4 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_4 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_4 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_4 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_4 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_4 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_4 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_4 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_4 + 57344));
                    unsigned int xw0_1_4 = rb0_1_4 + xoff0;
                    unsigned int xw1_1_4 = rb1_1_4 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_4));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_4 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_4 + 8192));
                    float dul_1_8 = __uint_as_float(wv0[0] << 16);
                    float duh_1_8 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_8[4];
                    float dyh_1_8[4];
                    dyl_1_8[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_8[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_8[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_8[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_8[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_8[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_8[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_8[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_16 = __uint_as_float(tv[1]);
                    float c0_1_16 = __uint_as_float(tv[2]);
                    float c1_1_16 = __uint_as_float(tv[3]);
                    float c2_1_16 = __uint_as_float(tv[4]);
                    float c3_1_16 = __uint_as_float(tv[5]);
                    float rl0_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_16) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load_0[0]));
                    float rh0_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_16) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load_0[1]));
                    float rl1_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_16) : "f"(pre_i_2_16), "f"(dul_1_8), "f"(rl0_1_16));
                    float rh1_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_16) : "f"(pre_i_2_16), "f"(duh_1_8), "f"(rh0_1_16));
                    float rl2_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_16) : "f"(c0_1_16), "f"(dyl_1_8[0]), "f"(rl1_1_16));
                    float rh2_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_16) : "f"(c0_1_16), "f"(dyh_1_8[0]), "f"(rh1_1_16));
                    float rl3_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_16) : "f"(c1_1_16), "f"(dyl_1_8[1]), "f"(rl2_1_16));
                    float rh3_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_16) : "f"(c1_1_16), "f"(dyh_1_8[1]), "f"(rh2_1_16));
                    float rl4_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_16) : "f"(c2_1_16), "f"(dyl_1_8[2]), "f"(rl3_1_16));
                    float rh4_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_16) : "f"(c2_1_16), "f"(dyh_1_8[2]), "f"(rh3_1_16));
                    float rl5_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_16) : "f"(c3_1_16), "f"(dyl_1_8[3]), "f"(rl4_1_16));
                    float rh5_1_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_16) : "f"(c3_1_16), "f"(dyh_1_8[3]), "f"(rh4_1_16));
                    unsigned int word_1_16 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_16, rh5_1_16)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_4), "r"(word_1_16) : "memory");
                    float pre_i_2_17 = __uint_as_float(tv[7]);
                    float c0_1_17 = __uint_as_float(tv[8]);
                    float c1_1_17 = __uint_as_float(tv[9]);
                    float c2_1_17 = __uint_as_float(tv[10]);
                    float c3_1_17 = __uint_as_float(tv[11]);
                    float rl0_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_17) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1_0[0]));
                    float rh0_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_17) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1_0[1]));
                    float rl1_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_17) : "f"(pre_i_2_17), "f"(dul_1_8), "f"(rl0_1_17));
                    float rh1_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_17) : "f"(pre_i_2_17), "f"(duh_1_8), "f"(rh0_1_17));
                    float rl2_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_17) : "f"(c0_1_17), "f"(dyl_1_8[0]), "f"(rl1_1_17));
                    float rh2_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_17) : "f"(c0_1_17), "f"(dyh_1_8[0]), "f"(rh1_1_17));
                    float rl3_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_17) : "f"(c1_1_17), "f"(dyl_1_8[1]), "f"(rl2_1_17));
                    float rh3_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_17) : "f"(c1_1_17), "f"(dyh_1_8[1]), "f"(rh2_1_17));
                    float rl4_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_17) : "f"(c2_1_17), "f"(dyl_1_8[2]), "f"(rl3_1_17));
                    float rh4_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_17) : "f"(c2_1_17), "f"(dyh_1_8[2]), "f"(rh3_1_17));
                    float rl5_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_17) : "f"(c3_1_17), "f"(dyl_1_8[3]), "f"(rl4_1_17));
                    float rh5_1_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_17) : "f"(c3_1_17), "f"(dyh_1_8[3]), "f"(rh4_1_17));
                    unsigned int word_1_17 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_17, rh5_1_17)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_4 + 8192), "r"(word_1_17) : "memory");
                    float dul_1_9 = __uint_as_float(wv1[0] << 16);
                    float duh_1_9 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_9[4];
                    float dyh_1_9[4];
                    dyl_1_9[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_9[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_9[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_9[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_9[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_9[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_9[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_9[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_18 = __uint_as_float(tv[13]);
                    float c0_1_18 = __uint_as_float(tv[14]);
                    float c1_1_18 = __uint_as_float(tv[15]);
                    float c2_1_18 = __uint_as_float(tv[16]);
                    float c3_1_18 = __uint_as_float(tv[17]);
                    float rl0_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_18) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load_0[2]));
                    float rh0_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_18) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load_0[3]));
                    float rl1_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_18) : "f"(pre_i_2_18), "f"(dul_1_9), "f"(rl0_1_18));
                    float rh1_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_18) : "f"(pre_i_2_18), "f"(duh_1_9), "f"(rh0_1_18));
                    float rl2_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_18) : "f"(c0_1_18), "f"(dyl_1_9[0]), "f"(rl1_1_18));
                    float rh2_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_18) : "f"(c0_1_18), "f"(dyh_1_9[0]), "f"(rh1_1_18));
                    float rl3_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_18) : "f"(c1_1_18), "f"(dyl_1_9[1]), "f"(rl2_1_18));
                    float rh3_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_18) : "f"(c1_1_18), "f"(dyh_1_9[1]), "f"(rh2_1_18));
                    float rl4_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_18) : "f"(c2_1_18), "f"(dyl_1_9[2]), "f"(rl3_1_18));
                    float rh4_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_18) : "f"(c2_1_18), "f"(dyh_1_9[2]), "f"(rh3_1_18));
                    float rl5_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_18) : "f"(c3_1_18), "f"(dyl_1_9[3]), "f"(rl4_1_18));
                    float rh5_1_18;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_18) : "f"(c3_1_18), "f"(dyh_1_9[3]), "f"(rh4_1_18));
                    unsigned int word_1_18 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_18, rh5_1_18)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_4), "r"(word_1_18) : "memory");
                    float pre_i_2_19 = __uint_as_float(tv[19]);
                    float c0_1_19 = __uint_as_float(tv[20]);
                    float c1_1_19 = __uint_as_float(tv[21]);
                    float c2_1_19 = __uint_as_float(tv[22]);
                    float c3_1_19 = __uint_as_float(tv[23]);
                    float rl0_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_19) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1_0[2]));
                    float rh0_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_19) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1_0[3]));
                    float rl1_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_19) : "f"(pre_i_2_19), "f"(dul_1_9), "f"(rl0_1_19));
                    float rh1_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_19) : "f"(pre_i_2_19), "f"(duh_1_9), "f"(rh0_1_19));
                    float rl2_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_19) : "f"(c0_1_19), "f"(dyl_1_9[0]), "f"(rl1_1_19));
                    float rh2_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_19) : "f"(c0_1_19), "f"(dyh_1_9[0]), "f"(rh1_1_19));
                    float rl3_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_19) : "f"(c1_1_19), "f"(dyl_1_9[1]), "f"(rl2_1_19));
                    float rh3_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_19) : "f"(c1_1_19), "f"(dyh_1_9[1]), "f"(rh2_1_19));
                    float rl4_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_19) : "f"(c2_1_19), "f"(dyl_1_9[2]), "f"(rl3_1_19));
                    float rh4_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_19) : "f"(c2_1_19), "f"(dyh_1_9[2]), "f"(rh3_1_19));
                    float rl5_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_19) : "f"(c3_1_19), "f"(dyl_1_9[3]), "f"(rl4_1_19));
                    float rh5_1_19;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_19) : "f"(c3_1_19), "f"(dyh_1_9[3]), "f"(rh4_1_19));
                    unsigned int word_1_19 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_19, rh5_1_19)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_4 + 8192), "r"(word_1_19) : "memory");
                    unsigned int rb0_1_5 = sbase + offg[5] + rowb0;
                    unsigned int rb1_1_5 = rb0_1_5 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_5 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_5 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_5 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_5 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_5 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_5 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_5 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_5 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_5 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_5 + 57344));
                    unsigned int xw0_1_5 = rb0_1_5 + xoff0;
                    unsigned int xw1_1_5 = rb1_1_5 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_5));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_5));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_5 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_5 + 8192));
                    float dul_1_10 = __uint_as_float(wv0[0] << 16);
                    float duh_1_10 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_10[4];
                    float dyh_1_10[4];
                    dyl_1_10[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_10[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_10[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_10[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_10[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_10[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_10[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_10[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_20 = __uint_as_float(tv[1]);
                    float c0_1_20 = __uint_as_float(tv[2]);
                    float c1_1_20 = __uint_as_float(tv[3]);
                    float c2_1_20 = __uint_as_float(tv[4]);
                    float c3_1_20 = __uint_as_float(tv[5]);
                    float rl0_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_20) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load_0[4]));
                    float rh0_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_20) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load_0[5]));
                    float rl1_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_20) : "f"(pre_i_2_20), "f"(dul_1_10), "f"(rl0_1_20));
                    float rh1_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_20) : "f"(pre_i_2_20), "f"(duh_1_10), "f"(rh0_1_20));
                    float rl2_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_20) : "f"(c0_1_20), "f"(dyl_1_10[0]), "f"(rl1_1_20));
                    float rh2_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_20) : "f"(c0_1_20), "f"(dyh_1_10[0]), "f"(rh1_1_20));
                    float rl3_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_20) : "f"(c1_1_20), "f"(dyl_1_10[1]), "f"(rl2_1_20));
                    float rh3_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_20) : "f"(c1_1_20), "f"(dyh_1_10[1]), "f"(rh2_1_20));
                    float rl4_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_20) : "f"(c2_1_20), "f"(dyl_1_10[2]), "f"(rl3_1_20));
                    float rh4_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_20) : "f"(c2_1_20), "f"(dyh_1_10[2]), "f"(rh3_1_20));
                    float rl5_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_20) : "f"(c3_1_20), "f"(dyl_1_10[3]), "f"(rl4_1_20));
                    float rh5_1_20;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_20) : "f"(c3_1_20), "f"(dyh_1_10[3]), "f"(rh4_1_20));
                    unsigned int word_1_20 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_20, rh5_1_20)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_5), "r"(word_1_20) : "memory");
                    float pre_i_2_21 = __uint_as_float(tv[7]);
                    float c0_1_21 = __uint_as_float(tv[8]);
                    float c1_1_21 = __uint_as_float(tv[9]);
                    float c2_1_21 = __uint_as_float(tv[10]);
                    float c3_1_21 = __uint_as_float(tv[11]);
                    float rl0_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_21) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1_0[4]));
                    float rh0_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_21) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1_0[5]));
                    float rl1_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_21) : "f"(pre_i_2_21), "f"(dul_1_10), "f"(rl0_1_21));
                    float rh1_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_21) : "f"(pre_i_2_21), "f"(duh_1_10), "f"(rh0_1_21));
                    float rl2_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_21) : "f"(c0_1_21), "f"(dyl_1_10[0]), "f"(rl1_1_21));
                    float rh2_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_21) : "f"(c0_1_21), "f"(dyh_1_10[0]), "f"(rh1_1_21));
                    float rl3_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_21) : "f"(c1_1_21), "f"(dyl_1_10[1]), "f"(rl2_1_21));
                    float rh3_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_21) : "f"(c1_1_21), "f"(dyh_1_10[1]), "f"(rh2_1_21));
                    float rl4_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_21) : "f"(c2_1_21), "f"(dyl_1_10[2]), "f"(rl3_1_21));
                    float rh4_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_21) : "f"(c2_1_21), "f"(dyh_1_10[2]), "f"(rh3_1_21));
                    float rl5_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_21) : "f"(c3_1_21), "f"(dyl_1_10[3]), "f"(rl4_1_21));
                    float rh5_1_21;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_21) : "f"(c3_1_21), "f"(dyh_1_10[3]), "f"(rh4_1_21));
                    unsigned int word_1_21 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_21, rh5_1_21)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_5 + 8192), "r"(word_1_21) : "memory");
                    float dul_1_11 = __uint_as_float(wv1[0] << 16);
                    float duh_1_11 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_11[4];
                    float dyh_1_11[4];
                    dyl_1_11[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_11[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_11[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_11[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_11[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_11[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_11[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_11[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_22 = __uint_as_float(tv[13]);
                    float c0_1_22 = __uint_as_float(tv[14]);
                    float c1_1_22 = __uint_as_float(tv[15]);
                    float c2_1_22 = __uint_as_float(tv[16]);
                    float c3_1_22 = __uint_as_float(tv[17]);
                    float rl0_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_22) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load_0[6]));
                    float rh0_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_22) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load_0[7]));
                    float rl1_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_22) : "f"(pre_i_2_22), "f"(dul_1_11), "f"(rl0_1_22));
                    float rh1_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_22) : "f"(pre_i_2_22), "f"(duh_1_11), "f"(rh0_1_22));
                    float rl2_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_22) : "f"(c0_1_22), "f"(dyl_1_11[0]), "f"(rl1_1_22));
                    float rh2_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_22) : "f"(c0_1_22), "f"(dyh_1_11[0]), "f"(rh1_1_22));
                    float rl3_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_22) : "f"(c1_1_22), "f"(dyl_1_11[1]), "f"(rl2_1_22));
                    float rh3_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_22) : "f"(c1_1_22), "f"(dyh_1_11[1]), "f"(rh2_1_22));
                    float rl4_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_22) : "f"(c2_1_22), "f"(dyl_1_11[2]), "f"(rl3_1_22));
                    float rh4_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_22) : "f"(c2_1_22), "f"(dyh_1_11[2]), "f"(rh3_1_22));
                    float rl5_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_22) : "f"(c3_1_22), "f"(dyl_1_11[3]), "f"(rl4_1_22));
                    float rh5_1_22;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_22) : "f"(c3_1_22), "f"(dyh_1_11[3]), "f"(rh4_1_22));
                    unsigned int word_1_22 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_22, rh5_1_22)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_5), "r"(word_1_22) : "memory");
                    float pre_i_2_23 = __uint_as_float(tv[19]);
                    float c0_1_23 = __uint_as_float(tv[20]);
                    float c1_1_23 = __uint_as_float(tv[21]);
                    float c2_1_23 = __uint_as_float(tv[22]);
                    float c3_1_23 = __uint_as_float(tv[23]);
                    float rl0_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_23) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1_0[6]));
                    float rh0_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_23) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1_0[7]));
                    float rl1_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_23) : "f"(pre_i_2_23), "f"(dul_1_11), "f"(rl0_1_23));
                    float rh1_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_23) : "f"(pre_i_2_23), "f"(duh_1_11), "f"(rh0_1_23));
                    float rl2_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_23) : "f"(c0_1_23), "f"(dyl_1_11[0]), "f"(rl1_1_23));
                    float rh2_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_23) : "f"(c0_1_23), "f"(dyh_1_11[0]), "f"(rh1_1_23));
                    float rl3_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_23) : "f"(c1_1_23), "f"(dyl_1_11[1]), "f"(rl2_1_23));
                    float rh3_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_23) : "f"(c1_1_23), "f"(dyh_1_11[1]), "f"(rh2_1_23));
                    float rl4_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_23) : "f"(c2_1_23), "f"(dyl_1_11[2]), "f"(rl3_1_23));
                    float rh4_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_23) : "f"(c2_1_23), "f"(dyh_1_11[2]), "f"(rh3_1_23));
                    float rl5_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_23) : "f"(c3_1_23), "f"(dyl_1_11[3]), "f"(rl4_1_23));
                    float rh5_1_23;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_23) : "f"(c3_1_23), "f"(dyh_1_11[3]), "f"(rh4_1_23));
                    unsigned int word_1_23 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_23, rh5_1_23)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_5 + 8192), "r"(word_1_23) : "memory");
                    unsigned int rb0_1_6 = sbase + offg[6] + rowb0;
                    unsigned int rb1_1_6 = rb0_1_6 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_6 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_6 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_6 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_6 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_6 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_6 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_6 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_6 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_6 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_6 + 57344));
                    unsigned int xw0_1_6 = rb0_1_6 + xoff0;
                    unsigned int xw1_1_6 = rb1_1_6 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_6));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_6));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_6 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_6 + 8192));
                    float dul_1_12 = __uint_as_float(wv0[0] << 16);
                    float duh_1_12 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_12[4];
                    float dyh_1_12[4];
                    dyl_1_12[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_12[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_12[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_12[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_12[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_12[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_12[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_12[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_24 = __uint_as_float(tv[1]);
                    float c0_1_24 = __uint_as_float(tv[2]);
                    float c1_1_24 = __uint_as_float(tv[3]);
                    float c2_1_24 = __uint_as_float(tv[4]);
                    float c3_1_24 = __uint_as_float(tv[5]);
                    float rl0_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_24) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load_0[8]));
                    float rh0_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_24) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load_0[9]));
                    float rl1_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_24) : "f"(pre_i_2_24), "f"(dul_1_12), "f"(rl0_1_24));
                    float rh1_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_24) : "f"(pre_i_2_24), "f"(duh_1_12), "f"(rh0_1_24));
                    float rl2_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_24) : "f"(c0_1_24), "f"(dyl_1_12[0]), "f"(rl1_1_24));
                    float rh2_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_24) : "f"(c0_1_24), "f"(dyh_1_12[0]), "f"(rh1_1_24));
                    float rl3_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_24) : "f"(c1_1_24), "f"(dyl_1_12[1]), "f"(rl2_1_24));
                    float rh3_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_24) : "f"(c1_1_24), "f"(dyh_1_12[1]), "f"(rh2_1_24));
                    float rl4_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_24) : "f"(c2_1_24), "f"(dyl_1_12[2]), "f"(rl3_1_24));
                    float rh4_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_24) : "f"(c2_1_24), "f"(dyh_1_12[2]), "f"(rh3_1_24));
                    float rl5_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_24) : "f"(c3_1_24), "f"(dyl_1_12[3]), "f"(rl4_1_24));
                    float rh5_1_24;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_24) : "f"(c3_1_24), "f"(dyh_1_12[3]), "f"(rh4_1_24));
                    unsigned int word_1_24 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_24, rh5_1_24)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_6), "r"(word_1_24) : "memory");
                    float pre_i_2_25 = __uint_as_float(tv[7]);
                    float c0_1_25 = __uint_as_float(tv[8]);
                    float c1_1_25 = __uint_as_float(tv[9]);
                    float c2_1_25 = __uint_as_float(tv[10]);
                    float c3_1_25 = __uint_as_float(tv[11]);
                    float rl0_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_25) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1_0[8]));
                    float rh0_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_25) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1_0[9]));
                    float rl1_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_25) : "f"(pre_i_2_25), "f"(dul_1_12), "f"(rl0_1_25));
                    float rh1_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_25) : "f"(pre_i_2_25), "f"(duh_1_12), "f"(rh0_1_25));
                    float rl2_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_25) : "f"(c0_1_25), "f"(dyl_1_12[0]), "f"(rl1_1_25));
                    float rh2_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_25) : "f"(c0_1_25), "f"(dyh_1_12[0]), "f"(rh1_1_25));
                    float rl3_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_25) : "f"(c1_1_25), "f"(dyl_1_12[1]), "f"(rl2_1_25));
                    float rh3_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_25) : "f"(c1_1_25), "f"(dyh_1_12[1]), "f"(rh2_1_25));
                    float rl4_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_25) : "f"(c2_1_25), "f"(dyl_1_12[2]), "f"(rl3_1_25));
                    float rh4_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_25) : "f"(c2_1_25), "f"(dyh_1_12[2]), "f"(rh3_1_25));
                    float rl5_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_25) : "f"(c3_1_25), "f"(dyl_1_12[3]), "f"(rl4_1_25));
                    float rh5_1_25;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_25) : "f"(c3_1_25), "f"(dyh_1_12[3]), "f"(rh4_1_25));
                    unsigned int word_1_25 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_25, rh5_1_25)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_6 + 8192), "r"(word_1_25) : "memory");
                    float dul_1_13 = __uint_as_float(wv1[0] << 16);
                    float duh_1_13 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_13[4];
                    float dyh_1_13[4];
                    dyl_1_13[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_13[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_13[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_13[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_13[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_13[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_13[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_13[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_26 = __uint_as_float(tv[13]);
                    float c0_1_26 = __uint_as_float(tv[14]);
                    float c1_1_26 = __uint_as_float(tv[15]);
                    float c2_1_26 = __uint_as_float(tv[16]);
                    float c3_1_26 = __uint_as_float(tv[17]);
                    float rl0_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_26) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load_0[10]));
                    float rh0_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_26) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load_0[11]));
                    float rl1_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_26) : "f"(pre_i_2_26), "f"(dul_1_13), "f"(rl0_1_26));
                    float rh1_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_26) : "f"(pre_i_2_26), "f"(duh_1_13), "f"(rh0_1_26));
                    float rl2_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_26) : "f"(c0_1_26), "f"(dyl_1_13[0]), "f"(rl1_1_26));
                    float rh2_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_26) : "f"(c0_1_26), "f"(dyh_1_13[0]), "f"(rh1_1_26));
                    float rl3_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_26) : "f"(c1_1_26), "f"(dyl_1_13[1]), "f"(rl2_1_26));
                    float rh3_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_26) : "f"(c1_1_26), "f"(dyh_1_13[1]), "f"(rh2_1_26));
                    float rl4_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_26) : "f"(c2_1_26), "f"(dyl_1_13[2]), "f"(rl3_1_26));
                    float rh4_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_26) : "f"(c2_1_26), "f"(dyh_1_13[2]), "f"(rh3_1_26));
                    float rl5_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_26) : "f"(c3_1_26), "f"(dyl_1_13[3]), "f"(rl4_1_26));
                    float rh5_1_26;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_26) : "f"(c3_1_26), "f"(dyh_1_13[3]), "f"(rh4_1_26));
                    unsigned int word_1_26 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_26, rh5_1_26)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_6), "r"(word_1_26) : "memory");
                    float pre_i_2_27 = __uint_as_float(tv[19]);
                    float c0_1_27 = __uint_as_float(tv[20]);
                    float c1_1_27 = __uint_as_float(tv[21]);
                    float c2_1_27 = __uint_as_float(tv[22]);
                    float c3_1_27 = __uint_as_float(tv[23]);
                    float rl0_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_27) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1_0[10]));
                    float rh0_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_27) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1_0[11]));
                    float rl1_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_27) : "f"(pre_i_2_27), "f"(dul_1_13), "f"(rl0_1_27));
                    float rh1_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_27) : "f"(pre_i_2_27), "f"(duh_1_13), "f"(rh0_1_27));
                    float rl2_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_27) : "f"(c0_1_27), "f"(dyl_1_13[0]), "f"(rl1_1_27));
                    float rh2_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_27) : "f"(c0_1_27), "f"(dyh_1_13[0]), "f"(rh1_1_27));
                    float rl3_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_27) : "f"(c1_1_27), "f"(dyl_1_13[1]), "f"(rl2_1_27));
                    float rh3_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_27) : "f"(c1_1_27), "f"(dyh_1_13[1]), "f"(rh2_1_27));
                    float rl4_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_27) : "f"(c2_1_27), "f"(dyl_1_13[2]), "f"(rl3_1_27));
                    float rh4_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_27) : "f"(c2_1_27), "f"(dyh_1_13[2]), "f"(rh3_1_27));
                    float rl5_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_27) : "f"(c3_1_27), "f"(dyl_1_13[3]), "f"(rl4_1_27));
                    float rh5_1_27;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_27) : "f"(c3_1_27), "f"(dyh_1_13[3]), "f"(rh4_1_27));
                    unsigned int word_1_27 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_27, rh5_1_27)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_6 + 8192), "r"(word_1_27) : "memory");
                    unsigned int rb0_1_7 = sbase + offg[7] + rowb0;
                    unsigned int rb1_1_7 = rb0_1_7 + 1024;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[0])) : "r"(rb0_1_7 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[0])) : "r"(rb1_1_7 + 65536));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[1])) : "r"(rb0_1_7 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[1])) : "r"(rb1_1_7 + 32768));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[2])) : "r"(rb0_1_7 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[2])) : "r"(rb1_1_7 + 40960));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[3])) : "r"(rb0_1_7 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[3])) : "r"(rb1_1_7 + 49152));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv0[4])) : "r"(rb0_1_7 + 57344));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv1[4])) : "r"(rb1_1_7 + 57344));
                    unsigned int xw0_1_7 = rb0_1_7 + xoff0;
                    unsigned int xw1_1_7 = rb1_1_7 + xoff0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[0])) : "r"(xw0_1_7));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[0])) : "r"(xw1_1_7));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr0[1])) : "r"(xw0_1_7 + 8192));
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xr1[1])) : "r"(xw1_1_7 + 8192));
                    float dul_1_14 = __uint_as_float(wv0[0] << 16);
                    float duh_1_14 = __uint_as_float(wv0[0] & 4294901760u);
                    float dyl_1_14[4];
                    float dyh_1_14[4];
                    dyl_1_14[0] = __uint_as_float(wv0[1] << 16);
                    dyh_1_14[0] = __uint_as_float(wv0[1] & 4294901760u);
                    dyl_1_14[1] = __uint_as_float(wv0[2] << 16);
                    dyh_1_14[1] = __uint_as_float(wv0[2] & 4294901760u);
                    dyl_1_14[2] = __uint_as_float(wv0[3] << 16);
                    dyh_1_14[2] = __uint_as_float(wv0[3] & 4294901760u);
                    dyl_1_14[3] = __uint_as_float(wv0[4] << 16);
                    dyh_1_14[3] = __uint_as_float(wv0[4] & 4294901760u);
                    float pre_i_2_28 = __uint_as_float(tv[1]);
                    float c0_1_28 = __uint_as_float(tv[2]);
                    float c1_1_28 = __uint_as_float(tv[3]);
                    float c2_1_28 = __uint_as_float(tv[4]);
                    float c3_1_28 = __uint_as_float(tv[5]);
                    float rl0_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_28) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(tmem_load_0[12]));
                    float rh0_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_28) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(tmem_load_0[13]));
                    float rl1_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_28) : "f"(pre_i_2_28), "f"(dul_1_14), "f"(rl0_1_28));
                    float rh1_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_28) : "f"(pre_i_2_28), "f"(duh_1_14), "f"(rh0_1_28));
                    float rl2_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_28) : "f"(c0_1_28), "f"(dyl_1_14[0]), "f"(rl1_1_28));
                    float rh2_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_28) : "f"(c0_1_28), "f"(dyh_1_14[0]), "f"(rh1_1_28));
                    float rl3_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_28) : "f"(c1_1_28), "f"(dyl_1_14[1]), "f"(rl2_1_28));
                    float rh3_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_28) : "f"(c1_1_28), "f"(dyh_1_14[1]), "f"(rh2_1_28));
                    float rl4_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_28) : "f"(c2_1_28), "f"(dyl_1_14[2]), "f"(rl3_1_28));
                    float rh4_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_28) : "f"(c2_1_28), "f"(dyh_1_14[2]), "f"(rh3_1_28));
                    float rl5_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_28) : "f"(c3_1_28), "f"(dyl_1_14[3]), "f"(rl4_1_28));
                    float rh5_1_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_28) : "f"(c3_1_28), "f"(dyh_1_14[3]), "f"(rh4_1_28));
                    unsigned int word_1_28 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_28, rh5_1_28)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_7), "r"(word_1_28) : "memory");
                    float pre_i_2_29 = __uint_as_float(tv[7]);
                    float c0_1_29 = __uint_as_float(tv[8]);
                    float c1_1_29 = __uint_as_float(tv[9]);
                    float c2_1_29 = __uint_as_float(tv[10]);
                    float c3_1_29 = __uint_as_float(tv[11]);
                    float rl0_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_29) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(tmem_load_1_0[12]));
                    float rh0_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_29) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(tmem_load_1_0[13]));
                    float rl1_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_29) : "f"(pre_i_2_29), "f"(dul_1_14), "f"(rl0_1_29));
                    float rh1_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_29) : "f"(pre_i_2_29), "f"(duh_1_14), "f"(rh0_1_29));
                    float rl2_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_29) : "f"(c0_1_29), "f"(dyl_1_14[0]), "f"(rl1_1_29));
                    float rh2_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_29) : "f"(c0_1_29), "f"(dyh_1_14[0]), "f"(rh1_1_29));
                    float rl3_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_29) : "f"(c1_1_29), "f"(dyl_1_14[1]), "f"(rl2_1_29));
                    float rh3_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_29) : "f"(c1_1_29), "f"(dyh_1_14[1]), "f"(rh2_1_29));
                    float rl4_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_29) : "f"(c2_1_29), "f"(dyl_1_14[2]), "f"(rl3_1_29));
                    float rh4_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_29) : "f"(c2_1_29), "f"(dyh_1_14[2]), "f"(rh3_1_29));
                    float rl5_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_29) : "f"(c3_1_29), "f"(dyl_1_14[3]), "f"(rl4_1_29));
                    float rh5_1_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_29) : "f"(c3_1_29), "f"(dyh_1_14[3]), "f"(rh4_1_29));
                    unsigned int word_1_29 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_29, rh5_1_29)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_1_7 + 8192), "r"(word_1_29) : "memory");
                    float dul_1_15 = __uint_as_float(wv1[0] << 16);
                    float duh_1_15 = __uint_as_float(wv1[0] & 4294901760u);
                    float dyl_1_15[4];
                    float dyh_1_15[4];
                    dyl_1_15[0] = __uint_as_float(wv1[1] << 16);
                    dyh_1_15[0] = __uint_as_float(wv1[1] & 4294901760u);
                    dyl_1_15[1] = __uint_as_float(wv1[2] << 16);
                    dyh_1_15[1] = __uint_as_float(wv1[2] & 4294901760u);
                    dyl_1_15[2] = __uint_as_float(wv1[3] << 16);
                    dyh_1_15[2] = __uint_as_float(wv1[3] & 4294901760u);
                    dyl_1_15[3] = __uint_as_float(wv1[4] << 16);
                    dyh_1_15[3] = __uint_as_float(wv1[4] & 4294901760u);
                    float pre_i_2_30 = __uint_as_float(tv[13]);
                    float c0_1_30 = __uint_as_float(tv[14]);
                    float c1_1_30 = __uint_as_float(tv[15]);
                    float c2_1_30 = __uint_as_float(tv[16]);
                    float c3_1_30 = __uint_as_float(tv[17]);
                    float rl0_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_30) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(tmem_load_0[14]));
                    float rh0_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_30) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(tmem_load_0[15]));
                    float rl1_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_30) : "f"(pre_i_2_30), "f"(dul_1_15), "f"(rl0_1_30));
                    float rh1_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_30) : "f"(pre_i_2_30), "f"(duh_1_15), "f"(rh0_1_30));
                    float rl2_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_30) : "f"(c0_1_30), "f"(dyl_1_15[0]), "f"(rl1_1_30));
                    float rh2_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_30) : "f"(c0_1_30), "f"(dyh_1_15[0]), "f"(rh1_1_30));
                    float rl3_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_30) : "f"(c1_1_30), "f"(dyl_1_15[1]), "f"(rl2_1_30));
                    float rh3_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_30) : "f"(c1_1_30), "f"(dyh_1_15[1]), "f"(rh2_1_30));
                    float rl4_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_30) : "f"(c2_1_30), "f"(dyl_1_15[2]), "f"(rl3_1_30));
                    float rh4_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_30) : "f"(c2_1_30), "f"(dyh_1_15[2]), "f"(rh3_1_30));
                    float rl5_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_30) : "f"(c3_1_30), "f"(dyl_1_15[3]), "f"(rl4_1_30));
                    float rh5_1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_30) : "f"(c3_1_30), "f"(dyh_1_15[3]), "f"(rh4_1_30));
                    unsigned int word_1_30 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_30, rh5_1_30)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_7), "r"(word_1_30) : "memory");
                    float pre_i_2_31 = __uint_as_float(tv[19]);
                    float c0_1_31 = __uint_as_float(tv[20]);
                    float c1_1_31 = __uint_as_float(tv[21]);
                    float c2_1_31 = __uint_as_float(tv[22]);
                    float c3_1_31 = __uint_as_float(tv[23]);
                    float rl0_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_1_31) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(tmem_load_1_0[14]));
                    float rh0_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_1_31) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(tmem_load_1_0[15]));
                    float rl1_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_1_31) : "f"(pre_i_2_31), "f"(dul_1_15), "f"(rl0_1_31));
                    float rh1_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_1_31) : "f"(pre_i_2_31), "f"(duh_1_15), "f"(rh0_1_31));
                    float rl2_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_1_31) : "f"(c0_1_31), "f"(dyl_1_15[0]), "f"(rl1_1_31));
                    float rh2_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_1_31) : "f"(c0_1_31), "f"(dyh_1_15[0]), "f"(rh1_1_31));
                    float rl3_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_1_31) : "f"(c1_1_31), "f"(dyl_1_15[1]), "f"(rl2_1_31));
                    float rh3_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_1_31) : "f"(c1_1_31), "f"(dyh_1_15[1]), "f"(rh2_1_31));
                    float rl4_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_1_31) : "f"(c2_1_31), "f"(dyl_1_15[2]), "f"(rl3_1_31));
                    float rh4_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_1_31) : "f"(c2_1_31), "f"(dyh_1_15[2]), "f"(rh3_1_31));
                    float rl5_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_1_31) : "f"(c3_1_31), "f"(dyl_1_15[3]), "f"(rl4_1_31));
                    float rh5_1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_1_31) : "f"(c3_1_31), "f"(dyh_1_15[3]), "f"(rh4_1_31));
                    unsigned int word_1_31 = __as_u32(__float22bfloat162_rn(make_float2(rl5_1_31, rh5_1_31)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_1_7 + 8192), "r"(word_1_31) : "memory");
                    int nxt = tile_2 + 1;
                    if (nxt < n_tiles_2) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (i0 == 0) {
                            #pragma unroll
                            for (int kq_4 = 0; kq_4 < 2; kq_4++) {
                                int rowa_2 = t0_1 + 64 + tokbase + wq_r4[kq_4];
                                if (rowa_2 >= T) {
                                    rowa_2 = T - 1;
                                }
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                    :: "r"(wtab + (unsigned int)(wq_r4[kq_4] * 80 + wq_q4[kq_4] * 16)), "l"(tokp + ((unsigned long long)rowa_2 * 24 + (unsigned long long)(wq_q4[kq_4] * 4))));
                            }
                        } else {
                            #pragma unroll
                            for (int kq_5 = 0; kq_5 < 3; kq_5++) {
                                int rowa_3 = t0_1 + 64 + tokbase + wq_r5[kq_5];
                                if (rowa_3 >= T) {
                                    rowa_3 = T - 1;
                                }
                                if (kq_5 == 2) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .pred p;\n\t"
                                        "setp.ne.b32 p, %0, 0;\n\t"
                                        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                        "}"
                                        :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_5] * 80 + wq_s5[kq_5] * 16)), "l"(tokp + ((unsigned long long)rowa_3 * 24 + (unsigned long long)(wq_q5[kq_5] * 4))));
                                } else {
                                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                        :: "r"(wtab + (unsigned int)(wq_r5[kq_5] * 80 + wq_s5[kq_5] * 16)), "l"(tokp + ((unsigned long long)rowa_3 * 24 + (unsigned long long)(wq_q5[kq_5] * 4))));
                                }
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                    } else {
                        int j1_1 = j_2 + 1;
                        if (j1_1 < n_mine_2) {
                            int u_1_1 = bid_0_2 + j1_1 * grid_2;
                            int s_1_1 = u_1_1 / P;
                            int p_1_1 = u_1_1 - s_1_1 * P;
                            int row0_1_1 = p_1_1 * rpc;
                            int rows_1_1 = T - row0_1_1;
                            if (rows_1_1 > rpc) {
                                rows_1_1 = rpc;
                            }
                            int n_tiles_1_1 = 0;
                            if (rows_1_1 > 0) {
                                n_tiles_1_1 = (rows_1_1 + 63) / 64;
                            }
                            if (n_tiles_1_1 > 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                __syncwarp();
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                if (i0 == 0) {
                                    #pragma unroll
                                    for (int kq_6 = 0; kq_6 < 2; kq_6++) {
                                        int rowa_4 = row0_1_1 + tokbase + wq_r4[kq_6];
                                        if (rowa_4 >= T) {
                                            rowa_4 = T - 1;
                                        }
                                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                            :: "r"(wtab + (unsigned int)(wq_r4[kq_6] * 80 + wq_q4[kq_6] * 16)), "l"(tokp + ((unsigned long long)rowa_4 * 24 + (unsigned long long)(wq_q4[kq_6] * 4))));
                                    }
                                } else {
                                    #pragma unroll
                                    for (int kq_7 = 0; kq_7 < 3; kq_7++) {
                                        int rowa_5 = row0_1_1 + tokbase + wq_r5[kq_7];
                                        if (rowa_5 >= T) {
                                            rowa_5 = T - 1;
                                        }
                                        if (kq_7 == 2) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .pred p;\n\t"
                                                "setp.ne.b32 p, %0, 0;\n\t"
                                                "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                                "}"
                                                :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_7] * 80 + wq_s5[kq_7] * 16)), "l"(tokp + ((unsigned long long)rowa_5 * 24 + (unsigned long long)(wq_q5[kq_7] * 4))));
                                        } else {
                                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                                :: "r"(wtab + (unsigned int)(wq_r5[kq_7] * 80 + wq_s5[kq_7] * 16)), "l"(tokp + ((unsigned long long)rowa_5 * 24 + (unsigned long long)(wq_q5[kq_7] * 4))));
                                        }
                                    }
                                }
                                asm volatile("cp.async.commit_group;");
                                qa_pre = 1;
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(stage_empty_e_addr + (ep_stage) * 8);
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
                        float sub;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub) : "f"(_tmem_load_4[n_1]), "f"(comp[n_1]));
                        float add;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(add) : "f"(acc[n_1]), "f"(sub));
                        float _sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(add), "f"(acc[n_1]));
                        float _sub_0;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_sub_1), "f"(sub));
                        comp[n_1] = _sub_0;
                        acc[n_1] = add;
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
                            float sub_1;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(sub_1) : "f"(_tmem_load_5[n_2]), "f"(comp[n_2]));
                            float add_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(add_1) : "f"(acc[n_2]), "f"(sub_1));
                            float _sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(add_1), "f"(acc[n_2]));
                            float _sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_sub_3), "f"(sub_1));
                            comp[n_2] = _sub_2;
                            acc[n_2] = add_1;
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
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tma_store_2d((&dX), s_2 * 64 + i0 * 4096, t0_1 + wcoord_t, sbase + xoff0 + wrow_bytes);
                        tma_store_2d((&dX), s_2 * 64 + (i0 + 1) * 4096, t0_1 + wcoord_t, sbase + xoff0 + 8192 + wrow_bytes);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(stage_empty_addr + (ep_stage) * 8);
                    }
                    ep_stage += 1;
                    if (ep_stage == 2) { ep_stage = 0; _phase_stage_full_x ^= 1; }
                }
                unsigned long long kcol = (unsigned long long)(i_str * 4096 + s_2 * 64 + dl);
                unsigned long long pbase = (unsigned long long)p_2 * 393216;
                #pragma unroll
                for (int n_3 = 0; n_3 < 24; n_3++) {
                    float _sub_4;
                    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(acc[n_3]), "f"(comp[n_3]));
                    dfn_part[pbase + (unsigned long long)(n_3 * 16384) + kcol] = _sub_4;
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
