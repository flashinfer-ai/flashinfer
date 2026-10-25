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
#define NUM_TK_PIPE_STAGES 2
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_WTV_OFF 222208
#define SMEM_WTV_STAGE_BYTES 6144
#define SMEM_WTV_STRIDE 6144
#define SMEM_TKV_OFF 228352
#define SMEM_TKV_STAGE_BYTES 16
#define SMEM_TKV_STRIDE 16
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
#define SMEM_TOTAL 228480
#define THREADS 384

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

__global__ void
kernel_cake_mhc_train_bwd_slab__dynr(const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap dY, const __grid_constant__ CUtensorMap dU, const __grid_constant__ CUtensorMap dzp, const __grid_constant__ CUtensorMap fnTp, const __grid_constant__ CUtensorMap dX, float* __restrict__ tokp, float* __restrict__ dfn_part, int T, int P, int rpc, unsigned int* __restrict__ ticket)
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
    #define tk_full_addr (mbar_base + 96)
    #define tk_empty_addr (mbar_base + 112)
    #define acc_full_addr (mbar_base + 128)
    #define acc_empty_addr (mbar_base + 144)
    #define dfn_full_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* wtv = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int wtv_addr = smem + 222208;
    unsigned int* tkv = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int tkv_addr = smem + 228352;
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

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

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
            // --- pipeline 'tk_pipe' ---
            // tk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tk_empty: 2 barriers, init_count=9
            mbarrier_init(smem + 112, 9);
            mbarrier_init(smem + 120, 9);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // acc_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 144, 8);
            mbarrier_init(smem + 152, 8);
            // dfn_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 0) {
        int _tmem_hold = smem + 176;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int _phase_tk_empty = 1;
            unsigned int _phase_fn_empty = 1;
            unsigned int _phase_stage_empty_e = 1;
            unsigned int _phase_stage_empty = 1;
            if (elect_sync()) {
                const int bid_0 = (int)bid;
                const int grid = (int)num_bids;
                const int n_units = 64 * P;
                const int n_mine = n_units + 1;
                const unsigned int tk_limit = (unsigned int)(n_units + grid - 1);
                unsigned int drawn = (unsigned int)bid_0;
                if (n_units > grid) {
                    uint32_t _atomic_fetch_inc_0;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(_atomic_fetch_inc_0) : "l"(ticket), "r"(static_cast<uint32_t>(tk_limit)) : "memory");
                    drawn = _atomic_fetch_inc_0;
                }
                unsigned int lk_stage = 0;
                unsigned int ld_stage = 0;
                int first_ld = 1;
                int tcl = 0;
                #pragma unroll 1
                for (int j = 0; j < n_mine; j++) {
                    mbarrier_wait(tk_empty_addr + (lk_stage) * 8, _phase_tk_empty);
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(tkv_addr + lk_stage * 4), "r"(drawn) : "memory");
                    mbarrier_arrive(tk_full_addr + (lk_stage) * 8);
                    lk_stage += 1;
                    if (lk_stage == 2) { lk_stage = 0; _phase_tk_empty ^= 1; }
                    int u = (int)drawn;
                    if (u >= n_units) {
                        break;
                    }
                    if (n_units <= grid) {
                        drawn = (unsigned int)(u + grid);
                    }
                    if (n_units > grid) {
                        uint32_t _atomic_fetch_inc_1;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_fetch_inc_1) : "l"(ticket), "r"(static_cast<uint32_t>(tk_limit)) : "memory");
                        drawn = _atomic_fetch_inc_1;
                    }
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
                        tma_3d_gmem2smem(fn0_0_addr, (&fnTp), 0, k0, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_0_addr + 4096, (&fnTp), 0, k0 + 4096, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_1_addr, (&fnTp), 0, k0 + 8192, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn0_1_addr + 4096, (&fnTp), 0, k0 + 12288, 0, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr, (&fnTp), 0, k0, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_0_addr + 4096, (&fnTp), 0, k0 + 4096, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_1_addr, (&fnTp), 0, k0 + 8192, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn1_1_addr + 4096, (&fnTp), 0, k0 + 12288, 1, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr, (&fnTp), 0, k0, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_0_addr + 4096, (&fnTp), 0, k0 + 4096, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_1_addr, (&fnTp), 0, k0 + 8192, 2, fn_full_addr);
                        tma_3d_gmem2smem(fn2_1_addr + 4096, (&fnTp), 0, k0 + 12288, 2, fn_full_addr);
                        mbarrier_arrive_expect_tx(fn_full_addr, 49152);
                        _phase_fn_empty ^= 1;
                        #pragma unroll 1
                        for (int tile = 0; tile < n_tiles; tile++) {
                            mbarrier_wait(stage_empty_e_addr + (ld_stage) * 8, _phase_stage_empty_e);
                            int t0 = row0 + tile * 64;
                            tma_2d_gmem2smem(dz0_addr + ld_stage * 86016, (&dzp), 0, t0, stage_full_z_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz1_addr + ld_stage * 86016, (&dzp), 32, t0, stage_full_z_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dz2_addr + ld_stage * 86016, (&dzp), 64, t0, stage_full_z_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_z_addr + (ld_stage) * 8, 12288);
                            tma_2d_gmem2smem(dus_addr + ld_stage * 86016, (&dU), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy0_addr + ld_stage * 86016, (&dY), k0, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy1_addr + ld_stage * 86016, (&dY), k0 + 4096, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy2_addr + ld_stage * 86016, (&dY), k0 + 8192, t0, stage_full_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(dy3_addr + ld_stage * 86016, (&dY), k0 + 12288, t0, stage_full_addr + (ld_stage) * 8);
                            mbarrier_arrive_expect_tx(stage_full_addr + (ld_stage) * 8, 40960);
                            mbarrier_wait(stage_empty_addr + (ld_stage) * 8, _phase_stage_empty);
                            tma_2d_gmem2smem(x0_addr + ld_stage * 86016, (&X), k0, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x1_addr + ld_stage * 86016, (&X), k0 + 4096, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x2_addr + ld_stage * 86016, (&X), k0 + 8192, t0, stage_full_x_addr + (ld_stage) * 8);
                            tma_2d_gmem2smem(x3_addr + ld_stage * 86016, (&X), k0 + 12288, t0, stage_full_x_addr + (ld_stage) * 8);
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
                                int j1 = (int)drawn;
                                if (j1 < n_units) {
                                    int s_0 = j1 / P;
                                    int p_1 = j1 - s_0 * P;
                                    int row0_2 = p_1 * rpc;
                                    int rows_3 = T - row0_2;
                                    if (rows_3 > rpc) {
                                        rows_3 = rpc;
                                    }
                                    int n_tiles_4 = 0;
                                    if (rows_3 > 0) {
                                        n_tiles_4 = (rows_3 + 63) / 64;
                                    }
                                    int tt = nt - n_tiles;
                                    if (tt < n_tiles_4) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 4096)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 4096)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 8192)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 8192)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&X))), "r"((int)(s_0 * 64 + 12288)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dY))), "r"((int)(s_0 * 64 + 12288)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dU))), "r"((int)(s_0 * 64)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(0)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(32)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&dzp))), "r"((int)(64)), "r"((int)(row0_2 + tt * 64)) : "memory");
                                    }
                                }
                            }
                            ld_stage += 1;
                            if (ld_stage == 2) { ld_stage = 0; _phase_stage_empty_e ^= 1; _phase_stage_empty ^= 1; }
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
            const int n_units_1 = 64 * P;
            const int n_mine_1 = n_units_1 + 1;
            unsigned int mk_stage = 0;
            unsigned int mm_stage = 0;
            unsigned int acc_mm = 0;
            unsigned int grp_mm = 0;
            int tcm = 0;
            unsigned int _phase_tk_full = 0;
            unsigned int _phase_fn_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_stage_full_z = 0;
            unsigned int _phase_stage_full_x = 0;
            #pragma unroll 1
            for (int j_1 = 0; j_1 < n_mine_1; j_1++) {
                mbarrier_wait(tk_full_addr + (mk_stage) * 8, _phase_tk_full);
                unsigned int tku[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku[0])) : "r"(tkv_addr + mk_stage * 4) : "memory");
                int u_1 = (int)tku[0];
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tk_empty_addr + (mk_stage) * 8);
                }
                mk_stage += 1;
                if (mk_stage == 2) { mk_stage = 0; _phase_tk_full ^= 1; }
                if (u_1 >= n_units_1) {
                    break;
                }
                int s_1 = u_1 / P;
                int p_2 = u_1 - s_1 * P;
                int row0_1 = p_2 * rpc;
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
                        int _mma_a_lo_0 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_0 = make_warp_uniform(((fnw2_addr) >> 4) & 0x3FFF);
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
                        int _mma_a_lo_1 = make_warp_uniform((((dz2_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_1 = make_warp_uniform(((fnw0_addr) >> 4) & 0x3FFF);
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
                        int _mma_a_lo_2 = make_warp_uniform((((dz1_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
                        int _mma_b_lo_2 = make_warp_uniform(((fnw1_addr) >> 4) & 0x3FFF);
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
                        int _mma_a_lo_3 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                        int _mma_a_lo_4 = make_warp_uniform((((dz1_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                        int _mma_a_lo_5 = make_warp_uniform((((dz0_addr) >> 4) & 0x3FFF) + (mm_stage) * 5376);
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
                        mbarrier_wait(stage_full_x_addr + (mm_stage) * 8, _phase_stage_full_x);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int in_grp = tile_1 % 16;
                        int small_init = ((in_grp == 0) ? 1 : 0);
                        unsigned int sbuf = grp_mm % 2;
                        int _mma_a_lo_6 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_3 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        int _mma_a_lo_7 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_4 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        int _mma_a_lo_8 = make_warp_uniform(((((xa0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_5 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        int _mma_a_lo_9 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_6 = make_warp_uniform(((((dzn0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        int _mma_a_lo_10 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_7 = make_warp_uniform(((((dzn1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        int _mma_a_lo_11 = make_warp_uniform(((((xa1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mm_stage) * 5376);
                        int _mma_b_lo_8 = make_warp_uniform(((((dzn2_addr) >> 4) & 0x3FFF) | 0x1000000) + (mm_stage) * 5376);
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
                        if (mm_stage == 2) { mm_stage = 0; _phase_stage_full_z ^= 1; _phase_stage_full_x ^= 1; }
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
            const int n_units_2 = 64 * P;
            const int n_mine_2 = n_units_2 + 1;
            unsigned int ek_stage = 0;
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
            const unsigned int wtab = wtv_addr + (unsigned int)(ew * 768);
            const unsigned int wpre_off = 8;
            const unsigned int wc_off = 16;
            const unsigned int wrow_bytes = (unsigned int)(tokbase * 128);
            const int wcoord_t = tokbase;
            int wq_r4[2];
            int wq_b4[2];
            #pragma unroll
            for (int kq = 0; kq < 2; kq++) {
                int cq = lane_1 + kq * 32;
                wq_r4[kq] = cq / 4;
                wq_b4[kq] = cq - cq / 4 * 4;
            }
            int wq_r5[3];
            int wq_b5[3];
            int wq_w5[3];
            #pragma unroll
            for (int kq_1 = 0; kq_1 < 3; kq_1++) {
                int cq_1 = lane_1 + kq_1 * 32;
                wq_r5[kq_1] = cq_1 / 5;
                wq_b5[kq_1] = cq_1 - cq_1 / 5 * 5;
                wq_w5[kq_1] = 4 + 8 * ((wq_b5[kq_1] + 4) / 5) + 2 * wq_b5[kq_1];
            }
            int qa_pre = 0;
            unsigned int _phase_tk_full_1 = 0;
            unsigned int _phase_acc_full = 0;
            unsigned int _phase_stage_full_x_1 = 0;
            unsigned int _phase_stage_full = 0;
            unsigned int _phase_dfn_full = 0;
            #pragma unroll 1
            for (int j_2 = 0; j_2 < n_mine_2; j_2++) {
                mbarrier_wait(tk_full_addr + (ek_stage) * 8, _phase_tk_full_1);
                unsigned int tku_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tku_1[0])) : "r"(tkv_addr + ek_stage * 4) : "memory");
                int u_2 = (int)tku_1[0];
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(tk_empty_addr + (ek_stage) * 8);
                }
                ek_stage += 1;
                if (ek_stage == 2) { ek_stage = 0; _phase_tk_full_1 ^= 1; }
                if (u_2 >= n_units_2) {
                    break;
                }
                int s_2 = u_2 / P;
                int p_3 = u_2 - s_2 * P;
                int row0_3 = p_3 * rpc;
                int rows_2 = T - row0_3;
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
                    int t0_1 = row0_3 + tile_2 * 64;
                    if (tile_2 == 0) {
                        if (qa_pre == 0) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            __syncwarp();
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int rowq = t0_1 + tokbase + lane_1;
                            if (rowq >= T) {
                                rowq = T - 1;
                            }
                            if (i0 == 0) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p;\n\t"
                                    "setp.ne.b32 p, %0, 0;\n\t"
                                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                    "}"
                                    :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48)), "l"(tokp + ((unsigned long long)rowq * 24)));
                                #pragma unroll
                                for (int kq_2 = 0; kq_2 < 2; kq_2++) {
                                    int rowa = t0_1 + tokbase + wq_r4[kq_2];
                                    if (rowa >= T) {
                                        rowa = T - 1;
                                    }
                                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                        :: "r"(wtab + (unsigned int)(wq_r4[kq_2] * 48 + 16 + 8 * wq_b4[kq_2])), "l"(tokp + ((unsigned long long)rowa * 24 + (unsigned long long)(6 + 2 * wq_b4[kq_2]))));
                                }
                            } else {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p;\n\t"
                                    "setp.ne.b32 p, %0, 0;\n\t"
                                    "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                                    "}"
                                    :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48 + 4)), "l"(tokp + ((unsigned long long)rowq * 24 + 1)));
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
                                            "@p cp.async.ca.shared::cta.global [%1], [%2], 8;\n\t"
                                            "}"
                                            :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_3] * 48 + 8 + 8 * wq_b5[kq_3])), "l"(tokp + ((unsigned long long)rowa_1 * 24 + (unsigned long long)wq_w5[kq_3])));
                                    } else {
                                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                            :: "r"(wtab + (unsigned int)(wq_r5[kq_3] * 48 + 8 + 8 * wq_b5[kq_3])), "l"(tokp + ((unsigned long long)rowa_1 * 24 + (unsigned long long)wq_w5[kq_3])));
                                    }
                                }
                            }
                            asm volatile("cp.async.commit_group;");
                            asm volatile("cp.async.commit_group;");
                        }
                        qa_pre = 0;
                    }
                    mbarrier_wait(acc_full_addr + (acc_ep) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int tm = lane_tm + acc_ep * 64;
                    unsigned int tab = dz0_addr + ep_stage * 86016;
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
                    asm volatile("cp.async.wait_group 0;");
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int sbase = x0_addr + ep_stage * 86016;
                    unsigned int tv[24];
                    unsigned int wpp[2];
                    unsigned int wrow = wtab + (unsigned int)(tq * 48);
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
                    unsigned int wrow_0 = wtab + (unsigned int)((tq + 8) * 48);
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
                    mbarrier_wait(stage_full_x_addr + (ep_stage) * 8, _phase_stage_full_x_1);
                    mbarrier_wait(stage_full_addr + (ep_stage) * 8, _phase_stage_full);
                    mbarrier_wait(dfn_full_addr + (acc_ep) * 8, _phase_dfn_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
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
                    float rl0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[0]));
                    float rh0;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[1]));
                    float rl1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1) : "f"(pre_i), "f"(dul), "f"(rl0));
                    float rh1;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1) : "f"(pre_i), "f"(duh), "f"(rh0));
                    float rl2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2) : "f"(c0), "f"(dyl[0]), "f"(rl1));
                    float rh2;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2) : "f"(c0), "f"(dyh[0]), "f"(rh1));
                    float rl3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3) : "f"(c1), "f"(dyl[1]), "f"(rl2));
                    float rh3;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3) : "f"(c1), "f"(dyh[1]), "f"(rh2));
                    float rl4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4) : "f"(c2), "f"(dyl[2]), "f"(rl3));
                    float rh4;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4) : "f"(c2), "f"(dyh[2]), "f"(rh3));
                    float rl5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5) : "f"(c3), "f"(dyl[3]), "f"(rl4));
                    float rh5;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5) : "f"(c3), "f"(dyh[3]), "f"(rh4));
                    unsigned int word = __as_u32(__float22bfloat162_rn(make_float2(rl5, rh5)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0), "r"(word) : "memory");
                    float pre_i_1 = __uint_as_float(tv[7]);
                    float c0_2 = __uint_as_float(tv[8]);
                    float c1_3 = __uint_as_float(tv[9]);
                    float c2_4 = __uint_as_float(tv[10]);
                    float c3_5 = __uint_as_float(tv[11]);
                    float rl0_6;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_6) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[0]));
                    float rh0_7;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_7) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[1]));
                    float rl1_8;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_8) : "f"(pre_i_1), "f"(dul), "f"(rl0_6));
                    float rh1_9;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_9) : "f"(pre_i_1), "f"(duh), "f"(rh0_7));
                    float rl2_10;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_10) : "f"(c0_2), "f"(dyl[0]), "f"(rl1_8));
                    float rh2_11;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_11) : "f"(c0_2), "f"(dyh[0]), "f"(rh1_9));
                    float rl3_12;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_12) : "f"(c1_3), "f"(dyl[1]), "f"(rl2_10));
                    float rh3_13;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_13) : "f"(c1_3), "f"(dyh[1]), "f"(rh2_11));
                    float rl4_14;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_14) : "f"(c2_4), "f"(dyl[2]), "f"(rl3_12));
                    float rh4_15;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_15) : "f"(c2_4), "f"(dyh[2]), "f"(rh3_13));
                    float rl5_16;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_16) : "f"(c3_5), "f"(dyl[3]), "f"(rl4_14));
                    float rh5_17;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_17) : "f"(c3_5), "f"(dyh[3]), "f"(rh4_15));
                    unsigned int word_18 = __as_u32(__float22bfloat162_rn(make_float2(rl5_16, rh5_17)));
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
                    float rl0_28;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_28) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[2]));
                    float rh0_29;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_29) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[3]));
                    float rl1_30;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_30) : "f"(pre_i_23), "f"(dul_19), "f"(rl0_28));
                    float rh1_31;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_31) : "f"(pre_i_23), "f"(duh_20), "f"(rh0_29));
                    float rl2_32;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_32) : "f"(c0_24), "f"(dyl_21[0]), "f"(rl1_30));
                    float rh2_33;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_33) : "f"(c0_24), "f"(dyh_22[0]), "f"(rh1_31));
                    float rl3_34;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_34) : "f"(c1_25), "f"(dyl_21[1]), "f"(rl2_32));
                    float rh3_35;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_35) : "f"(c1_25), "f"(dyh_22[1]), "f"(rh2_33));
                    float rl4_36;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_36) : "f"(c2_26), "f"(dyl_21[2]), "f"(rl3_34));
                    float rh4_37;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_37) : "f"(c2_26), "f"(dyh_22[2]), "f"(rh3_35));
                    float rl5_38;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_38) : "f"(c3_27), "f"(dyl_21[3]), "f"(rl4_36));
                    float rh5_39;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_39) : "f"(c3_27), "f"(dyh_22[3]), "f"(rh4_37));
                    unsigned int word_40 = __as_u32(__float22bfloat162_rn(make_float2(rl5_38, rh5_39)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1), "r"(word_40) : "memory");
                    float pre_i_41 = __uint_as_float(tv[19]);
                    float c0_42 = __uint_as_float(tv[20]);
                    float c1_43 = __uint_as_float(tv[21]);
                    float c2_44 = __uint_as_float(tv[22]);
                    float c3_45 = __uint_as_float(tv[23]);
                    float rl0_46;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_46) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[2]));
                    float rh0_47;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_47) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[3]));
                    float rl1_48;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_48) : "f"(pre_i_41), "f"(dul_19), "f"(rl0_46));
                    float rh1_49;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_49) : "f"(pre_i_41), "f"(duh_20), "f"(rh0_47));
                    float rl2_50;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_50) : "f"(c0_42), "f"(dyl_21[0]), "f"(rl1_48));
                    float rh2_51;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_51) : "f"(c0_42), "f"(dyh_22[0]), "f"(rh1_49));
                    float rl3_52;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_52) : "f"(c1_43), "f"(dyl_21[1]), "f"(rl2_50));
                    float rh3_53;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_53) : "f"(c1_43), "f"(dyh_22[1]), "f"(rh2_51));
                    float rl4_54;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_54) : "f"(c2_44), "f"(dyl_21[2]), "f"(rl3_52));
                    float rh4_55;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_55) : "f"(c2_44), "f"(dyh_22[2]), "f"(rh3_53));
                    float rl5_56;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_56) : "f"(c3_45), "f"(dyl_21[3]), "f"(rl4_54));
                    float rh5_57;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_57) : "f"(c3_45), "f"(dyh_22[3]), "f"(rh4_55));
                    unsigned int word_58 = __as_u32(__float22bfloat162_rn(make_float2(rl5_56, rh5_57)));
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
                    float rl0_72;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_72) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[4]));
                    float rh0_73;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_73) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[5]));
                    float rl1_74;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_74) : "f"(pre_i_67), "f"(dul_63), "f"(rl0_72));
                    float rh1_75;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_75) : "f"(pre_i_67), "f"(duh_64), "f"(rh0_73));
                    float rl2_76;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_76) : "f"(c0_68), "f"(dyl_65[0]), "f"(rl1_74));
                    float rh2_77;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_77) : "f"(c0_68), "f"(dyh_66[0]), "f"(rh1_75));
                    float rl3_78;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_78) : "f"(c1_69), "f"(dyl_65[1]), "f"(rl2_76));
                    float rh3_79;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_79) : "f"(c1_69), "f"(dyh_66[1]), "f"(rh2_77));
                    float rl4_80;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_80) : "f"(c2_70), "f"(dyl_65[2]), "f"(rl3_78));
                    float rh4_81;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_81) : "f"(c2_70), "f"(dyh_66[2]), "f"(rh3_79));
                    float rl5_82;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_82) : "f"(c3_71), "f"(dyl_65[3]), "f"(rl4_80));
                    float rh5_83;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_83) : "f"(c3_71), "f"(dyh_66[3]), "f"(rh4_81));
                    unsigned int word_84 = __as_u32(__float22bfloat162_rn(make_float2(rl5_82, rh5_83)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_61), "r"(word_84) : "memory");
                    float pre_i_85 = __uint_as_float(tv[7]);
                    float c0_86 = __uint_as_float(tv[8]);
                    float c1_87 = __uint_as_float(tv[9]);
                    float c2_88 = __uint_as_float(tv[10]);
                    float c3_89 = __uint_as_float(tv[11]);
                    float rl0_90;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_90) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[4]));
                    float rh0_91;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_91) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[5]));
                    float rl1_92;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_92) : "f"(pre_i_85), "f"(dul_63), "f"(rl0_90));
                    float rh1_93;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_93) : "f"(pre_i_85), "f"(duh_64), "f"(rh0_91));
                    float rl2_94;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_94) : "f"(c0_86), "f"(dyl_65[0]), "f"(rl1_92));
                    float rh2_95;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_95) : "f"(c0_86), "f"(dyh_66[0]), "f"(rh1_93));
                    float rl3_96;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_96) : "f"(c1_87), "f"(dyl_65[1]), "f"(rl2_94));
                    float rh3_97;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_97) : "f"(c1_87), "f"(dyh_66[1]), "f"(rh2_95));
                    float rl4_98;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_98) : "f"(c2_88), "f"(dyl_65[2]), "f"(rl3_96));
                    float rh4_99;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_99) : "f"(c2_88), "f"(dyh_66[2]), "f"(rh3_97));
                    float rl5_100;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_100) : "f"(c3_89), "f"(dyl_65[3]), "f"(rl4_98));
                    float rh5_101;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_101) : "f"(c3_89), "f"(dyh_66[3]), "f"(rh4_99));
                    unsigned int word_102 = __as_u32(__float22bfloat162_rn(make_float2(rl5_100, rh5_101)));
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
                    float rl0_112;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_112) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[6]));
                    float rh0_113;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_113) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[7]));
                    float rl1_114;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_114) : "f"(pre_i_107), "f"(dul_103), "f"(rl0_112));
                    float rh1_115;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_115) : "f"(pre_i_107), "f"(duh_104), "f"(rh0_113));
                    float rl2_116;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_116) : "f"(c0_108), "f"(dyl_105[0]), "f"(rl1_114));
                    float rh2_117;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_117) : "f"(c0_108), "f"(dyh_106[0]), "f"(rh1_115));
                    float rl3_118;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_118) : "f"(c1_109), "f"(dyl_105[1]), "f"(rl2_116));
                    float rh3_119;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_119) : "f"(c1_109), "f"(dyh_106[1]), "f"(rh2_117));
                    float rl4_120;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_120) : "f"(c2_110), "f"(dyl_105[2]), "f"(rl3_118));
                    float rh4_121;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_121) : "f"(c2_110), "f"(dyh_106[2]), "f"(rh3_119));
                    float rl5_122;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_122) : "f"(c3_111), "f"(dyl_105[3]), "f"(rl4_120));
                    float rh5_123;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_123) : "f"(c3_111), "f"(dyh_106[3]), "f"(rh4_121));
                    unsigned int word_124 = __as_u32(__float22bfloat162_rn(make_float2(rl5_122, rh5_123)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_62), "r"(word_124) : "memory");
                    float pre_i_125 = __uint_as_float(tv[19]);
                    float c0_126 = __uint_as_float(tv[20]);
                    float c1_127 = __uint_as_float(tv[21]);
                    float c2_128 = __uint_as_float(tv[22]);
                    float c3_129 = __uint_as_float(tv[23]);
                    float rl0_130;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_130) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[6]));
                    float rh0_131;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_131) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[7]));
                    float rl1_132;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_132) : "f"(pre_i_125), "f"(dul_103), "f"(rl0_130));
                    float rh1_133;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_133) : "f"(pre_i_125), "f"(duh_104), "f"(rh0_131));
                    float rl2_134;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_134) : "f"(c0_126), "f"(dyl_105[0]), "f"(rl1_132));
                    float rh2_135;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_135) : "f"(c0_126), "f"(dyh_106[0]), "f"(rh1_133));
                    float rl3_136;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_136) : "f"(c1_127), "f"(dyl_105[1]), "f"(rl2_134));
                    float rh3_137;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_137) : "f"(c1_127), "f"(dyh_106[1]), "f"(rh2_135));
                    float rl4_138;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_138) : "f"(c2_128), "f"(dyl_105[2]), "f"(rl3_136));
                    float rh4_139;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_139) : "f"(c2_128), "f"(dyh_106[2]), "f"(rh3_137));
                    float rl5_140;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_140) : "f"(c3_129), "f"(dyl_105[3]), "f"(rl4_138));
                    float rh5_141;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_141) : "f"(c3_129), "f"(dyh_106[3]), "f"(rh4_139));
                    unsigned int word_142 = __as_u32(__float22bfloat162_rn(make_float2(rl5_140, rh5_141)));
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
                    float rl0_156;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_156) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[8]));
                    float rh0_157;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_157) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[9]));
                    float rl1_158;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_158) : "f"(pre_i_151), "f"(dul_147), "f"(rl0_156));
                    float rh1_159;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_159) : "f"(pre_i_151), "f"(duh_148), "f"(rh0_157));
                    float rl2_160;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_160) : "f"(c0_152), "f"(dyl_149[0]), "f"(rl1_158));
                    float rh2_161;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_161) : "f"(c0_152), "f"(dyh_150[0]), "f"(rh1_159));
                    float rl3_162;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_162) : "f"(c1_153), "f"(dyl_149[1]), "f"(rl2_160));
                    float rh3_163;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_163) : "f"(c1_153), "f"(dyh_150[1]), "f"(rh2_161));
                    float rl4_164;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_164) : "f"(c2_154), "f"(dyl_149[2]), "f"(rl3_162));
                    float rh4_165;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_165) : "f"(c2_154), "f"(dyh_150[2]), "f"(rh3_163));
                    float rl5_166;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_166) : "f"(c3_155), "f"(dyl_149[3]), "f"(rl4_164));
                    float rh5_167;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_167) : "f"(c3_155), "f"(dyh_150[3]), "f"(rh4_165));
                    unsigned int word_168 = __as_u32(__float22bfloat162_rn(make_float2(rl5_166, rh5_167)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_145), "r"(word_168) : "memory");
                    float pre_i_169 = __uint_as_float(tv[7]);
                    float c0_170 = __uint_as_float(tv[8]);
                    float c1_171 = __uint_as_float(tv[9]);
                    float c2_172 = __uint_as_float(tv[10]);
                    float c3_173 = __uint_as_float(tv[11]);
                    float rl0_174;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_174) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[8]));
                    float rh0_175;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_175) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[9]));
                    float rl1_176;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_176) : "f"(pre_i_169), "f"(dul_147), "f"(rl0_174));
                    float rh1_177;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_177) : "f"(pre_i_169), "f"(duh_148), "f"(rh0_175));
                    float rl2_178;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_178) : "f"(c0_170), "f"(dyl_149[0]), "f"(rl1_176));
                    float rh2_179;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_179) : "f"(c0_170), "f"(dyh_150[0]), "f"(rh1_177));
                    float rl3_180;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_180) : "f"(c1_171), "f"(dyl_149[1]), "f"(rl2_178));
                    float rh3_181;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_181) : "f"(c1_171), "f"(dyh_150[1]), "f"(rh2_179));
                    float rl4_182;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_182) : "f"(c2_172), "f"(dyl_149[2]), "f"(rl3_180));
                    float rh4_183;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_183) : "f"(c2_172), "f"(dyh_150[2]), "f"(rh3_181));
                    float rl5_184;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_184) : "f"(c3_173), "f"(dyl_149[3]), "f"(rl4_182));
                    float rh5_185;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_185) : "f"(c3_173), "f"(dyh_150[3]), "f"(rh4_183));
                    unsigned int word_186 = __as_u32(__float22bfloat162_rn(make_float2(rl5_184, rh5_185)));
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
                    float rl0_196;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_196) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[10]));
                    float rh0_197;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_197) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[11]));
                    float rl1_198;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_198) : "f"(pre_i_191), "f"(dul_187), "f"(rl0_196));
                    float rh1_199;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_199) : "f"(pre_i_191), "f"(duh_188), "f"(rh0_197));
                    float rl2_200;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_200) : "f"(c0_192), "f"(dyl_189[0]), "f"(rl1_198));
                    float rh2_201;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_201) : "f"(c0_192), "f"(dyh_190[0]), "f"(rh1_199));
                    float rl3_202;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_202) : "f"(c1_193), "f"(dyl_189[1]), "f"(rl2_200));
                    float rh3_203;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_203) : "f"(c1_193), "f"(dyh_190[1]), "f"(rh2_201));
                    float rl4_204;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_204) : "f"(c2_194), "f"(dyl_189[2]), "f"(rl3_202));
                    float rh4_205;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_205) : "f"(c2_194), "f"(dyh_190[2]), "f"(rh3_203));
                    float rl5_206;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_206) : "f"(c3_195), "f"(dyl_189[3]), "f"(rl4_204));
                    float rh5_207;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_207) : "f"(c3_195), "f"(dyh_190[3]), "f"(rh4_205));
                    unsigned int word_208 = __as_u32(__float22bfloat162_rn(make_float2(rl5_206, rh5_207)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_146), "r"(word_208) : "memory");
                    float pre_i_209 = __uint_as_float(tv[19]);
                    float c0_210 = __uint_as_float(tv[20]);
                    float c1_211 = __uint_as_float(tv[21]);
                    float c2_212 = __uint_as_float(tv[22]);
                    float c3_213 = __uint_as_float(tv[23]);
                    float rl0_214;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_214) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[10]));
                    float rh0_215;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_215) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[11]));
                    float rl1_216;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_216) : "f"(pre_i_209), "f"(dul_187), "f"(rl0_214));
                    float rh1_217;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_217) : "f"(pre_i_209), "f"(duh_188), "f"(rh0_215));
                    float rl2_218;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_218) : "f"(c0_210), "f"(dyl_189[0]), "f"(rl1_216));
                    float rh2_219;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_219) : "f"(c0_210), "f"(dyh_190[0]), "f"(rh1_217));
                    float rl3_220;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_220) : "f"(c1_211), "f"(dyl_189[1]), "f"(rl2_218));
                    float rh3_221;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_221) : "f"(c1_211), "f"(dyh_190[1]), "f"(rh2_219));
                    float rl4_222;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_222) : "f"(c2_212), "f"(dyl_189[2]), "f"(rl3_220));
                    float rh4_223;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_223) : "f"(c2_212), "f"(dyh_190[2]), "f"(rh3_221));
                    float rl5_224;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_224) : "f"(c3_213), "f"(dyl_189[3]), "f"(rl4_222));
                    float rh5_225;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_225) : "f"(c3_213), "f"(dyh_190[3]), "f"(rh4_223));
                    unsigned int word_226 = __as_u32(__float22bfloat162_rn(make_float2(rl5_224, rh5_225)));
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
                    float rl0_240;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_240) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_0[12]));
                    float rh0_241;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_241) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_0[13]));
                    float rl1_242;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_242) : "f"(pre_i_235), "f"(dul_231), "f"(rl0_240));
                    float rh1_243;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_243) : "f"(pre_i_235), "f"(duh_232), "f"(rh0_241));
                    float rl2_244;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_244) : "f"(c0_236), "f"(dyl_233[0]), "f"(rl1_242));
                    float rh2_245;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_245) : "f"(c0_236), "f"(dyh_234[0]), "f"(rh1_243));
                    float rl3_246;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_246) : "f"(c1_237), "f"(dyl_233[1]), "f"(rl2_244));
                    float rh3_247;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_247) : "f"(c1_237), "f"(dyh_234[1]), "f"(rh2_245));
                    float rl4_248;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_248) : "f"(c2_238), "f"(dyl_233[2]), "f"(rl3_246));
                    float rh4_249;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_249) : "f"(c2_238), "f"(dyh_234[2]), "f"(rh3_247));
                    float rl5_250;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_250) : "f"(c3_239), "f"(dyl_233[3]), "f"(rl4_248));
                    float rh5_251;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_251) : "f"(c3_239), "f"(dyh_234[3]), "f"(rh4_249));
                    unsigned int word_252 = __as_u32(__float22bfloat162_rn(make_float2(rl5_250, rh5_251)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_229), "r"(word_252) : "memory");
                    float pre_i_253 = __uint_as_float(tv[7]);
                    float c0_254 = __uint_as_float(tv[8]);
                    float c1_255 = __uint_as_float(tv[9]);
                    float c2_256 = __uint_as_float(tv[10]);
                    float c3_257 = __uint_as_float(tv[11]);
                    float rl0_258;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_258) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_1[12]));
                    float rh0_259;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_259) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_1[13]));
                    float rl1_260;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_260) : "f"(pre_i_253), "f"(dul_231), "f"(rl0_258));
                    float rh1_261;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_261) : "f"(pre_i_253), "f"(duh_232), "f"(rh0_259));
                    float rl2_262;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_262) : "f"(c0_254), "f"(dyl_233[0]), "f"(rl1_260));
                    float rh2_263;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_263) : "f"(c0_254), "f"(dyh_234[0]), "f"(rh1_261));
                    float rl3_264;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_264) : "f"(c1_255), "f"(dyl_233[1]), "f"(rl2_262));
                    float rh3_265;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_265) : "f"(c1_255), "f"(dyh_234[1]), "f"(rh2_263));
                    float rl4_266;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_266) : "f"(c2_256), "f"(dyl_233[2]), "f"(rl3_264));
                    float rh4_267;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_267) : "f"(c2_256), "f"(dyh_234[2]), "f"(rh3_265));
                    float rl5_268;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_268) : "f"(c3_257), "f"(dyl_233[3]), "f"(rl4_266));
                    float rh5_269;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_269) : "f"(c3_257), "f"(dyh_234[3]), "f"(rh4_267));
                    unsigned int word_270 = __as_u32(__float22bfloat162_rn(make_float2(rl5_268, rh5_269)));
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
                    float rl0_280;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_280) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_0[14]));
                    float rh0_281;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_281) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_0[15]));
                    float rl1_282;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_282) : "f"(pre_i_275), "f"(dul_271), "f"(rl0_280));
                    float rh1_283;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_283) : "f"(pre_i_275), "f"(duh_272), "f"(rh0_281));
                    float rl2_284;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_284) : "f"(c0_276), "f"(dyl_273[0]), "f"(rl1_282));
                    float rh2_285;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_285) : "f"(c0_276), "f"(dyh_274[0]), "f"(rh1_283));
                    float rl3_286;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_286) : "f"(c1_277), "f"(dyl_273[1]), "f"(rl2_284));
                    float rh3_287;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_287) : "f"(c1_277), "f"(dyh_274[1]), "f"(rh2_285));
                    float rl4_288;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_288) : "f"(c2_278), "f"(dyl_273[2]), "f"(rl3_286));
                    float rh4_289;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_289) : "f"(c2_278), "f"(dyh_274[2]), "f"(rh3_287));
                    float rl5_290;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_290) : "f"(c3_279), "f"(dyl_273[3]), "f"(rl4_288));
                    float rh5_291;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_291) : "f"(c3_279), "f"(dyh_274[3]), "f"(rh4_289));
                    unsigned int word_292 = __as_u32(__float22bfloat162_rn(make_float2(rl5_290, rh5_291)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_230), "r"(word_292) : "memory");
                    float pre_i_293 = __uint_as_float(tv[19]);
                    float c0_294 = __uint_as_float(tv[20]);
                    float c1_295 = __uint_as_float(tv[21]);
                    float c2_296 = __uint_as_float(tv[22]);
                    float c3_297 = __uint_as_float(tv[23]);
                    float rl0_298;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_298) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_1[14]));
                    float rh0_299;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_299) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_1[15]));
                    float rl1_300;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_300) : "f"(pre_i_293), "f"(dul_271), "f"(rl0_298));
                    float rh1_301;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_301) : "f"(pre_i_293), "f"(duh_272), "f"(rh0_299));
                    float rl2_302;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_302) : "f"(c0_294), "f"(dyl_273[0]), "f"(rl1_300));
                    float rh2_303;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_303) : "f"(c0_294), "f"(dyh_274[0]), "f"(rh1_301));
                    float rl3_304;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_304) : "f"(c1_295), "f"(dyl_273[1]), "f"(rl2_302));
                    float rh3_305;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_305) : "f"(c1_295), "f"(dyh_274[1]), "f"(rh2_303));
                    float rl4_306;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_306) : "f"(c2_296), "f"(dyl_273[2]), "f"(rl3_304));
                    float rh4_307;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_307) : "f"(c2_296), "f"(dyh_274[2]), "f"(rh3_305));
                    float rl5_308;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_308) : "f"(c3_297), "f"(dyl_273[3]), "f"(rl4_306));
                    float rh5_309;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_309) : "f"(c3_297), "f"(dyh_274[3]), "f"(rh4_307));
                    unsigned int word_310 = __as_u32(__float22bfloat162_rn(make_float2(rl5_308, rh5_309)));
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
                    float rl0_324;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_324) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[0]));
                    float rh0_325;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_325) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[1]));
                    float rl1_326;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_326) : "f"(pre_i_319), "f"(dul_315), "f"(rl0_324));
                    float rh1_327;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_327) : "f"(pre_i_319), "f"(duh_316), "f"(rh0_325));
                    float rl2_328;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_328) : "f"(c0_320), "f"(dyl_317[0]), "f"(rl1_326));
                    float rh2_329;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_329) : "f"(c0_320), "f"(dyh_318[0]), "f"(rh1_327));
                    float rl3_330;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_330) : "f"(c1_321), "f"(dyl_317[1]), "f"(rl2_328));
                    float rh3_331;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_331) : "f"(c1_321), "f"(dyh_318[1]), "f"(rh2_329));
                    float rl4_332;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_332) : "f"(c2_322), "f"(dyl_317[2]), "f"(rl3_330));
                    float rh4_333;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_333) : "f"(c2_322), "f"(dyh_318[2]), "f"(rh3_331));
                    float rl5_334;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_334) : "f"(c3_323), "f"(dyl_317[3]), "f"(rl4_332));
                    float rh5_335;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_335) : "f"(c3_323), "f"(dyh_318[3]), "f"(rh4_333));
                    unsigned int word_336 = __as_u32(__float22bfloat162_rn(make_float2(rl5_334, rh5_335)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_313), "r"(word_336) : "memory");
                    float pre_i_337 = __uint_as_float(tv[7]);
                    float c0_338 = __uint_as_float(tv[8]);
                    float c1_339 = __uint_as_float(tv[9]);
                    float c2_340 = __uint_as_float(tv[10]);
                    float c3_341 = __uint_as_float(tv[11]);
                    float rl0_342;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_342) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[0]));
                    float rh0_343;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_343) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[1]));
                    float rl1_344;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_344) : "f"(pre_i_337), "f"(dul_315), "f"(rl0_342));
                    float rh1_345;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_345) : "f"(pre_i_337), "f"(duh_316), "f"(rh0_343));
                    float rl2_346;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_346) : "f"(c0_338), "f"(dyl_317[0]), "f"(rl1_344));
                    float rh2_347;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_347) : "f"(c0_338), "f"(dyh_318[0]), "f"(rh1_345));
                    float rl3_348;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_348) : "f"(c1_339), "f"(dyl_317[1]), "f"(rl2_346));
                    float rh3_349;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_349) : "f"(c1_339), "f"(dyh_318[1]), "f"(rh2_347));
                    float rl4_350;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_350) : "f"(c2_340), "f"(dyl_317[2]), "f"(rl3_348));
                    float rh4_351;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_351) : "f"(c2_340), "f"(dyh_318[2]), "f"(rh3_349));
                    float rl5_352;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_352) : "f"(c3_341), "f"(dyl_317[3]), "f"(rl4_350));
                    float rh5_353;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_353) : "f"(c3_341), "f"(dyh_318[3]), "f"(rh4_351));
                    unsigned int word_354 = __as_u32(__float22bfloat162_rn(make_float2(rl5_352, rh5_353)));
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
                    float rl0_364;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_364) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[2]));
                    float rh0_365;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_365) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[3]));
                    float rl1_366;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_366) : "f"(pre_i_359), "f"(dul_355), "f"(rl0_364));
                    float rh1_367;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_367) : "f"(pre_i_359), "f"(duh_356), "f"(rh0_365));
                    float rl2_368;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_368) : "f"(c0_360), "f"(dyl_357[0]), "f"(rl1_366));
                    float rh2_369;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_369) : "f"(c0_360), "f"(dyh_358[0]), "f"(rh1_367));
                    float rl3_370;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_370) : "f"(c1_361), "f"(dyl_357[1]), "f"(rl2_368));
                    float rh3_371;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_371) : "f"(c1_361), "f"(dyh_358[1]), "f"(rh2_369));
                    float rl4_372;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_372) : "f"(c2_362), "f"(dyl_357[2]), "f"(rl3_370));
                    float rh4_373;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_373) : "f"(c2_362), "f"(dyh_358[2]), "f"(rh3_371));
                    float rl5_374;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_374) : "f"(c3_363), "f"(dyl_357[3]), "f"(rl4_372));
                    float rh5_375;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_375) : "f"(c3_363), "f"(dyh_358[3]), "f"(rh4_373));
                    unsigned int word_376 = __as_u32(__float22bfloat162_rn(make_float2(rl5_374, rh5_375)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_314), "r"(word_376) : "memory");
                    float pre_i_377 = __uint_as_float(tv[19]);
                    float c0_378 = __uint_as_float(tv[20]);
                    float c1_379 = __uint_as_float(tv[21]);
                    float c2_380 = __uint_as_float(tv[22]);
                    float c3_381 = __uint_as_float(tv[23]);
                    float rl0_382;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_382) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[2]));
                    float rh0_383;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_383) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[3]));
                    float rl1_384;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_384) : "f"(pre_i_377), "f"(dul_355), "f"(rl0_382));
                    float rh1_385;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_385) : "f"(pre_i_377), "f"(duh_356), "f"(rh0_383));
                    float rl2_386;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_386) : "f"(c0_378), "f"(dyl_357[0]), "f"(rl1_384));
                    float rh2_387;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_387) : "f"(c0_378), "f"(dyh_358[0]), "f"(rh1_385));
                    float rl3_388;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_388) : "f"(c1_379), "f"(dyl_357[1]), "f"(rl2_386));
                    float rh3_389;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_389) : "f"(c1_379), "f"(dyh_358[1]), "f"(rh2_387));
                    float rl4_390;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_390) : "f"(c2_380), "f"(dyl_357[2]), "f"(rl3_388));
                    float rh4_391;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_391) : "f"(c2_380), "f"(dyh_358[2]), "f"(rh3_389));
                    float rl5_392;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_392) : "f"(c3_381), "f"(dyl_357[3]), "f"(rl4_390));
                    float rh5_393;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_393) : "f"(c3_381), "f"(dyh_358[3]), "f"(rh4_391));
                    unsigned int word_394 = __as_u32(__float22bfloat162_rn(make_float2(rl5_392, rh5_393)));
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
                    float rl0_408;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_408) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[4]));
                    float rh0_409;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_409) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[5]));
                    float rl1_410;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_410) : "f"(pre_i_403), "f"(dul_399), "f"(rl0_408));
                    float rh1_411;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_411) : "f"(pre_i_403), "f"(duh_400), "f"(rh0_409));
                    float rl2_412;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_412) : "f"(c0_404), "f"(dyl_401[0]), "f"(rl1_410));
                    float rh2_413;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_413) : "f"(c0_404), "f"(dyh_402[0]), "f"(rh1_411));
                    float rl3_414;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_414) : "f"(c1_405), "f"(dyl_401[1]), "f"(rl2_412));
                    float rh3_415;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_415) : "f"(c1_405), "f"(dyh_402[1]), "f"(rh2_413));
                    float rl4_416;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_416) : "f"(c2_406), "f"(dyl_401[2]), "f"(rl3_414));
                    float rh4_417;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_417) : "f"(c2_406), "f"(dyh_402[2]), "f"(rh3_415));
                    float rl5_418;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_418) : "f"(c3_407), "f"(dyl_401[3]), "f"(rl4_416));
                    float rh5_419;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_419) : "f"(c3_407), "f"(dyh_402[3]), "f"(rh4_417));
                    unsigned int word_420 = __as_u32(__float22bfloat162_rn(make_float2(rl5_418, rh5_419)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_397), "r"(word_420) : "memory");
                    float pre_i_421 = __uint_as_float(tv[7]);
                    float c0_422 = __uint_as_float(tv[8]);
                    float c1_423 = __uint_as_float(tv[9]);
                    float c2_424 = __uint_as_float(tv[10]);
                    float c3_425 = __uint_as_float(tv[11]);
                    float rl0_426;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_426) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[4]));
                    float rh0_427;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_427) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[5]));
                    float rl1_428;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_428) : "f"(pre_i_421), "f"(dul_399), "f"(rl0_426));
                    float rh1_429;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_429) : "f"(pre_i_421), "f"(duh_400), "f"(rh0_427));
                    float rl2_430;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_430) : "f"(c0_422), "f"(dyl_401[0]), "f"(rl1_428));
                    float rh2_431;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_431) : "f"(c0_422), "f"(dyh_402[0]), "f"(rh1_429));
                    float rl3_432;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_432) : "f"(c1_423), "f"(dyl_401[1]), "f"(rl2_430));
                    float rh3_433;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_433) : "f"(c1_423), "f"(dyh_402[1]), "f"(rh2_431));
                    float rl4_434;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_434) : "f"(c2_424), "f"(dyl_401[2]), "f"(rl3_432));
                    float rh4_435;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_435) : "f"(c2_424), "f"(dyh_402[2]), "f"(rh3_433));
                    float rl5_436;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_436) : "f"(c3_425), "f"(dyl_401[3]), "f"(rl4_434));
                    float rh5_437;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_437) : "f"(c3_425), "f"(dyh_402[3]), "f"(rh4_435));
                    unsigned int word_438 = __as_u32(__float22bfloat162_rn(make_float2(rl5_436, rh5_437)));
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
                    float rl0_448;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_448) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[6]));
                    float rh0_449;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_449) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[7]));
                    float rl1_450;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_450) : "f"(pre_i_443), "f"(dul_439), "f"(rl0_448));
                    float rh1_451;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_451) : "f"(pre_i_443), "f"(duh_440), "f"(rh0_449));
                    float rl2_452;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_452) : "f"(c0_444), "f"(dyl_441[0]), "f"(rl1_450));
                    float rh2_453;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_453) : "f"(c0_444), "f"(dyh_442[0]), "f"(rh1_451));
                    float rl3_454;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_454) : "f"(c1_445), "f"(dyl_441[1]), "f"(rl2_452));
                    float rh3_455;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_455) : "f"(c1_445), "f"(dyh_442[1]), "f"(rh2_453));
                    float rl4_456;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_456) : "f"(c2_446), "f"(dyl_441[2]), "f"(rl3_454));
                    float rh4_457;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_457) : "f"(c2_446), "f"(dyh_442[2]), "f"(rh3_455));
                    float rl5_458;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_458) : "f"(c3_447), "f"(dyl_441[3]), "f"(rl4_456));
                    float rh5_459;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_459) : "f"(c3_447), "f"(dyh_442[3]), "f"(rh4_457));
                    unsigned int word_460 = __as_u32(__float22bfloat162_rn(make_float2(rl5_458, rh5_459)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_398), "r"(word_460) : "memory");
                    float pre_i_461 = __uint_as_float(tv[19]);
                    float c0_462 = __uint_as_float(tv[20]);
                    float c1_463 = __uint_as_float(tv[21]);
                    float c2_464 = __uint_as_float(tv[22]);
                    float c3_465 = __uint_as_float(tv[23]);
                    float rl0_466;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_466) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[6]));
                    float rh0_467;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_467) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[7]));
                    float rl1_468;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_468) : "f"(pre_i_461), "f"(dul_439), "f"(rl0_466));
                    float rh1_469;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_469) : "f"(pre_i_461), "f"(duh_440), "f"(rh0_467));
                    float rl2_470;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_470) : "f"(c0_462), "f"(dyl_441[0]), "f"(rl1_468));
                    float rh2_471;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_471) : "f"(c0_462), "f"(dyh_442[0]), "f"(rh1_469));
                    float rl3_472;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_472) : "f"(c1_463), "f"(dyl_441[1]), "f"(rl2_470));
                    float rh3_473;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_473) : "f"(c1_463), "f"(dyh_442[1]), "f"(rh2_471));
                    float rl4_474;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_474) : "f"(c2_464), "f"(dyl_441[2]), "f"(rl3_472));
                    float rh4_475;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_475) : "f"(c2_464), "f"(dyh_442[2]), "f"(rh3_473));
                    float rl5_476;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_476) : "f"(c3_465), "f"(dyl_441[3]), "f"(rl4_474));
                    float rh5_477;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_477) : "f"(c3_465), "f"(dyh_442[3]), "f"(rh4_475));
                    unsigned int word_478 = __as_u32(__float22bfloat162_rn(make_float2(rl5_476, rh5_477)));
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
                    float rl0_492;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_492) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[8]));
                    float rh0_493;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_493) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[9]));
                    float rl1_494;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_494) : "f"(pre_i_487), "f"(dul_483), "f"(rl0_492));
                    float rh1_495;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_495) : "f"(pre_i_487), "f"(duh_484), "f"(rh0_493));
                    float rl2_496;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_496) : "f"(c0_488), "f"(dyl_485[0]), "f"(rl1_494));
                    float rh2_497;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_497) : "f"(c0_488), "f"(dyh_486[0]), "f"(rh1_495));
                    float rl3_498;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_498) : "f"(c1_489), "f"(dyl_485[1]), "f"(rl2_496));
                    float rh3_499;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_499) : "f"(c1_489), "f"(dyh_486[1]), "f"(rh2_497));
                    float rl4_500;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_500) : "f"(c2_490), "f"(dyl_485[2]), "f"(rl3_498));
                    float rh4_501;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_501) : "f"(c2_490), "f"(dyh_486[2]), "f"(rh3_499));
                    float rl5_502;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_502) : "f"(c3_491), "f"(dyl_485[3]), "f"(rl4_500));
                    float rh5_503;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_503) : "f"(c3_491), "f"(dyh_486[3]), "f"(rh4_501));
                    unsigned int word_504 = __as_u32(__float22bfloat162_rn(make_float2(rl5_502, rh5_503)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_481), "r"(word_504) : "memory");
                    float pre_i_505 = __uint_as_float(tv[7]);
                    float c0_506 = __uint_as_float(tv[8]);
                    float c1_507 = __uint_as_float(tv[9]);
                    float c2_508 = __uint_as_float(tv[10]);
                    float c3_509 = __uint_as_float(tv[11]);
                    float rl0_510;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_510) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[8]));
                    float rh0_511;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_511) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[9]));
                    float rl1_512;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_512) : "f"(pre_i_505), "f"(dul_483), "f"(rl0_510));
                    float rh1_513;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_513) : "f"(pre_i_505), "f"(duh_484), "f"(rh0_511));
                    float rl2_514;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_514) : "f"(c0_506), "f"(dyl_485[0]), "f"(rl1_512));
                    float rh2_515;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_515) : "f"(c0_506), "f"(dyh_486[0]), "f"(rh1_513));
                    float rl3_516;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_516) : "f"(c1_507), "f"(dyl_485[1]), "f"(rl2_514));
                    float rh3_517;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_517) : "f"(c1_507), "f"(dyh_486[1]), "f"(rh2_515));
                    float rl4_518;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_518) : "f"(c2_508), "f"(dyl_485[2]), "f"(rl3_516));
                    float rh4_519;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_519) : "f"(c2_508), "f"(dyh_486[2]), "f"(rh3_517));
                    float rl5_520;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_520) : "f"(c3_509), "f"(dyl_485[3]), "f"(rl4_518));
                    float rh5_521;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_521) : "f"(c3_509), "f"(dyh_486[3]), "f"(rh4_519));
                    unsigned int word_522 = __as_u32(__float22bfloat162_rn(make_float2(rl5_520, rh5_521)));
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
                    float rl0_532;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_532) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[10]));
                    float rh0_533;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_533) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[11]));
                    float rl1_534;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_534) : "f"(pre_i_527), "f"(dul_523), "f"(rl0_532));
                    float rh1_535;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_535) : "f"(pre_i_527), "f"(duh_524), "f"(rh0_533));
                    float rl2_536;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_536) : "f"(c0_528), "f"(dyl_525[0]), "f"(rl1_534));
                    float rh2_537;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_537) : "f"(c0_528), "f"(dyh_526[0]), "f"(rh1_535));
                    float rl3_538;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_538) : "f"(c1_529), "f"(dyl_525[1]), "f"(rl2_536));
                    float rh3_539;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_539) : "f"(c1_529), "f"(dyh_526[1]), "f"(rh2_537));
                    float rl4_540;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_540) : "f"(c2_530), "f"(dyl_525[2]), "f"(rl3_538));
                    float rh4_541;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_541) : "f"(c2_530), "f"(dyh_526[2]), "f"(rh3_539));
                    float rl5_542;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_542) : "f"(c3_531), "f"(dyl_525[3]), "f"(rl4_540));
                    float rh5_543;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_543) : "f"(c3_531), "f"(dyh_526[3]), "f"(rh4_541));
                    unsigned int word_544 = __as_u32(__float22bfloat162_rn(make_float2(rl5_542, rh5_543)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_482), "r"(word_544) : "memory");
                    float pre_i_545 = __uint_as_float(tv[19]);
                    float c0_546 = __uint_as_float(tv[20]);
                    float c1_547 = __uint_as_float(tv[21]);
                    float c2_548 = __uint_as_float(tv[22]);
                    float c3_549 = __uint_as_float(tv[23]);
                    float rl0_550;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_550) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[10]));
                    float rh0_551;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_551) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[11]));
                    float rl1_552;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_552) : "f"(pre_i_545), "f"(dul_523), "f"(rl0_550));
                    float rh1_553;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_553) : "f"(pre_i_545), "f"(duh_524), "f"(rh0_551));
                    float rl2_554;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_554) : "f"(c0_546), "f"(dyl_525[0]), "f"(rl1_552));
                    float rh2_555;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_555) : "f"(c0_546), "f"(dyh_526[0]), "f"(rh1_553));
                    float rl3_556;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_556) : "f"(c1_547), "f"(dyl_525[1]), "f"(rl2_554));
                    float rh3_557;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_557) : "f"(c1_547), "f"(dyh_526[1]), "f"(rh2_555));
                    float rl4_558;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_558) : "f"(c2_548), "f"(dyl_525[2]), "f"(rl3_556));
                    float rh4_559;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_559) : "f"(c2_548), "f"(dyh_526[2]), "f"(rh3_557));
                    float rl5_560;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_560) : "f"(c3_549), "f"(dyl_525[3]), "f"(rl4_558));
                    float rh5_561;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_561) : "f"(c3_549), "f"(dyh_526[3]), "f"(rh4_559));
                    unsigned int word_562 = __as_u32(__float22bfloat162_rn(make_float2(rl5_560, rh5_561)));
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
                    float rl0_576;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_576) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] << 16)), "f"(_tmem_load_2[12]));
                    float rh0_577;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_577) : "f"(nct[0]), "f"(__uint_as_float(xr0[0] & 4294901760u)), "f"(_tmem_load_2[13]));
                    float rl1_578;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_578) : "f"(pre_i_571), "f"(dul_567), "f"(rl0_576));
                    float rh1_579;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_579) : "f"(pre_i_571), "f"(duh_568), "f"(rh0_577));
                    float rl2_580;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_580) : "f"(c0_572), "f"(dyl_569[0]), "f"(rl1_578));
                    float rh2_581;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_581) : "f"(c0_572), "f"(dyh_570[0]), "f"(rh1_579));
                    float rl3_582;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_582) : "f"(c1_573), "f"(dyl_569[1]), "f"(rl2_580));
                    float rh3_583;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_583) : "f"(c1_573), "f"(dyh_570[1]), "f"(rh2_581));
                    float rl4_584;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_584) : "f"(c2_574), "f"(dyl_569[2]), "f"(rl3_582));
                    float rh4_585;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_585) : "f"(c2_574), "f"(dyh_570[2]), "f"(rh3_583));
                    float rl5_586;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_586) : "f"(c3_575), "f"(dyl_569[3]), "f"(rl4_584));
                    float rh5_587;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_587) : "f"(c3_575), "f"(dyh_570[3]), "f"(rh4_585));
                    unsigned int word_588 = __as_u32(__float22bfloat162_rn(make_float2(rl5_586, rh5_587)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw0_565), "r"(word_588) : "memory");
                    float pre_i_589 = __uint_as_float(tv[7]);
                    float c0_590 = __uint_as_float(tv[8]);
                    float c1_591 = __uint_as_float(tv[9]);
                    float c2_592 = __uint_as_float(tv[10]);
                    float c3_593 = __uint_as_float(tv[11]);
                    float rl0_594;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_594) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] << 16)), "f"(_tmem_load_3[12]));
                    float rh0_595;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_595) : "f"(nct[0]), "f"(__uint_as_float(xr0[1] & 4294901760u)), "f"(_tmem_load_3[13]));
                    float rl1_596;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_596) : "f"(pre_i_589), "f"(dul_567), "f"(rl0_594));
                    float rh1_597;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_597) : "f"(pre_i_589), "f"(duh_568), "f"(rh0_595));
                    float rl2_598;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_598) : "f"(c0_590), "f"(dyl_569[0]), "f"(rl1_596));
                    float rh2_599;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_599) : "f"(c0_590), "f"(dyh_570[0]), "f"(rh1_597));
                    float rl3_600;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_600) : "f"(c1_591), "f"(dyl_569[1]), "f"(rl2_598));
                    float rh3_601;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_601) : "f"(c1_591), "f"(dyh_570[1]), "f"(rh2_599));
                    float rl4_602;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_602) : "f"(c2_592), "f"(dyl_569[2]), "f"(rl3_600));
                    float rh4_603;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_603) : "f"(c2_592), "f"(dyh_570[2]), "f"(rh3_601));
                    float rl5_604;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_604) : "f"(c3_593), "f"(dyl_569[3]), "f"(rl4_602));
                    float rh5_605;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_605) : "f"(c3_593), "f"(dyh_570[3]), "f"(rh4_603));
                    unsigned int word_606 = __as_u32(__float22bfloat162_rn(make_float2(rl5_604, rh5_605)));
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
                    float rl0_616;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_616) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] << 16)), "f"(_tmem_load_2[14]));
                    float rh0_617;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_617) : "f"(nct[1]), "f"(__uint_as_float(xr1[0] & 4294901760u)), "f"(_tmem_load_2[15]));
                    float rl1_618;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_618) : "f"(pre_i_611), "f"(dul_607), "f"(rl0_616));
                    float rh1_619;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_619) : "f"(pre_i_611), "f"(duh_608), "f"(rh0_617));
                    float rl2_620;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_620) : "f"(c0_612), "f"(dyl_609[0]), "f"(rl1_618));
                    float rh2_621;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_621) : "f"(c0_612), "f"(dyh_610[0]), "f"(rh1_619));
                    float rl3_622;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_622) : "f"(c1_613), "f"(dyl_609[1]), "f"(rl2_620));
                    float rh3_623;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_623) : "f"(c1_613), "f"(dyh_610[1]), "f"(rh2_621));
                    float rl4_624;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_624) : "f"(c2_614), "f"(dyl_609[2]), "f"(rl3_622));
                    float rh4_625;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_625) : "f"(c2_614), "f"(dyh_610[2]), "f"(rh3_623));
                    float rl5_626;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_626) : "f"(c3_615), "f"(dyl_609[3]), "f"(rl4_624));
                    float rh5_627;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_627) : "f"(c3_615), "f"(dyh_610[3]), "f"(rh4_625));
                    unsigned int word_628 = __as_u32(__float22bfloat162_rn(make_float2(rl5_626, rh5_627)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_566), "r"(word_628) : "memory");
                    float pre_i_629 = __uint_as_float(tv[19]);
                    float c0_630 = __uint_as_float(tv[20]);
                    float c1_631 = __uint_as_float(tv[21]);
                    float c2_632 = __uint_as_float(tv[22]);
                    float c3_633 = __uint_as_float(tv[23]);
                    float rl0_634;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl0_634) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] << 16)), "f"(_tmem_load_3[14]));
                    float rh0_635;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh0_635) : "f"(nct[1]), "f"(__uint_as_float(xr1[1] & 4294901760u)), "f"(_tmem_load_3[15]));
                    float rl1_636;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl1_636) : "f"(pre_i_629), "f"(dul_607), "f"(rl0_634));
                    float rh1_637;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh1_637) : "f"(pre_i_629), "f"(duh_608), "f"(rh0_635));
                    float rl2_638;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl2_638) : "f"(c0_630), "f"(dyl_609[0]), "f"(rl1_636));
                    float rh2_639;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh2_639) : "f"(c0_630), "f"(dyh_610[0]), "f"(rh1_637));
                    float rl3_640;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl3_640) : "f"(c1_631), "f"(dyl_609[1]), "f"(rl2_638));
                    float rh3_641;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh3_641) : "f"(c1_631), "f"(dyh_610[1]), "f"(rh2_639));
                    float rl4_642;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl4_642) : "f"(c2_632), "f"(dyl_609[2]), "f"(rl3_640));
                    float rh4_643;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh4_643) : "f"(c2_632), "f"(dyh_610[2]), "f"(rh3_641));
                    float rl5_644;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rl5_644) : "f"(c3_633), "f"(dyl_609[3]), "f"(rl4_642));
                    float rh5_645;
                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(rh5_645) : "f"(c3_633), "f"(dyh_610[3]), "f"(rh4_643));
                    unsigned int word_646 = __as_u32(__float22bfloat162_rn(make_float2(rl5_644, rh5_645)));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(xw1_566 + 8192), "r"(word_646) : "memory");
                    int nxt = tile_2 + 1;
                    if (nxt < n_tiles_2) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int rowq_1 = t0_1 + 64 + tokbase + lane_1;
                        if (rowq_1 >= T) {
                            rowq_1 = T - 1;
                        }
                        if (i0 == 0) {
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                "}"
                                :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48)), "l"(tokp + ((unsigned long long)rowq_1 * 24)));
                            #pragma unroll
                            for (int kq_4 = 0; kq_4 < 2; kq_4++) {
                                int rowa_2 = t0_1 + 64 + tokbase + wq_r4[kq_4];
                                if (rowa_2 >= T) {
                                    rowa_2 = T - 1;
                                }
                                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                    :: "r"(wtab + (unsigned int)(wq_r4[kq_4] * 48 + 16 + 8 * wq_b4[kq_4])), "l"(tokp + ((unsigned long long)rowa_2 * 24 + (unsigned long long)(6 + 2 * wq_b4[kq_4]))));
                            }
                        } else {
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                "setp.ne.b32 p, %0, 0;\n\t"
                                "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                                "}"
                                :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48 + 4)), "l"(tokp + ((unsigned long long)rowq_1 * 24 + 1)));
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
                                        "@p cp.async.ca.shared::cta.global [%1], [%2], 8;\n\t"
                                        "}"
                                        :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_5] * 48 + 8 + 8 * wq_b5[kq_5])), "l"(tokp + ((unsigned long long)rowa_3 * 24 + (unsigned long long)wq_w5[kq_5])));
                                } else {
                                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                        :: "r"(wtab + (unsigned int)(wq_r5[kq_5] * 48 + 8 + 8 * wq_b5[kq_5])), "l"(tokp + ((unsigned long long)rowa_3 * 24 + (unsigned long long)wq_w5[kq_5])));
                                }
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.commit_group;");
                    } else {
                        mbarrier_wait(tk_full_addr + (ek_stage) * 8, _phase_tk_full_1);
                        unsigned int tkn[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&tkn[0])) : "r"(tkv_addr + ek_stage * 4) : "memory");
                        int j1_1 = (int)tkn[0];
                        if (j1_1 < n_units_2) {
                            int s_0_1 = j1_1 / P;
                            int p_1_1 = j1_1 - s_0_1 * P;
                            int row0_2_1 = p_1_1 * rpc;
                            int rows_3_1 = T - row0_2_1;
                            if (rows_3_1 > rpc) {
                                rows_3_1 = rpc;
                            }
                            int n_tiles_4_1 = 0;
                            if (rows_3_1 > 0) {
                                n_tiles_4_1 = (rows_3_1 + 63) / 64;
                            }
                            if (n_tiles_4_1 > 0) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                __syncwarp();
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int rowq_2 = row0_2_1 + tokbase + lane_1;
                                if (rowq_2 >= T) {
                                    rowq_2 = T - 1;
                                }
                                if (i0 == 0) {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .pred p;\n\t"
                                        "setp.ne.b32 p, %0, 0;\n\t"
                                        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                                        "}"
                                        :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48)), "l"(tokp + ((unsigned long long)rowq_2 * 24)));
                                    #pragma unroll
                                    for (int kq_6 = 0; kq_6 < 2; kq_6++) {
                                        int rowa_4 = row0_2_1 + tokbase + wq_r4[kq_6];
                                        if (rowa_4 >= T) {
                                            rowa_4 = T - 1;
                                        }
                                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                            :: "r"(wtab + (unsigned int)(wq_r4[kq_6] * 48 + 16 + 8 * wq_b4[kq_6])), "l"(tokp + ((unsigned long long)rowa_4 * 24 + (unsigned long long)(6 + 2 * wq_b4[kq_6]))));
                                    }
                                } else {
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .pred p;\n\t"
                                        "setp.ne.b32 p, %0, 0;\n\t"
                                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                                        "}"
                                        :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(lane_1 * 48 + 4)), "l"(tokp + ((unsigned long long)rowq_2 * 24 + 1)));
                                    #pragma unroll
                                    for (int kq_7 = 0; kq_7 < 3; kq_7++) {
                                        int rowa_5 = row0_2_1 + tokbase + wq_r5[kq_7];
                                        if (rowa_5 >= T) {
                                            rowa_5 = T - 1;
                                        }
                                        if (kq_7 == 2) {
                                            asm volatile(
                                                "{\n\t"
                                                ".reg .pred p;\n\t"
                                                "setp.ne.b32 p, %0, 0;\n\t"
                                                "@p cp.async.ca.shared::cta.global [%1], [%2], 8;\n\t"
                                                "}"
                                                :: "r"((lane_1 < 16) ? 1 : 0), "r"(wtab + (unsigned int)(wq_r5[kq_7] * 48 + 8 + 8 * wq_b5[kq_7])), "l"(tokp + ((unsigned long long)rowa_5 * 24 + (unsigned long long)wq_w5[kq_7])));
                                        } else {
                                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                                                :: "r"(wtab + (unsigned int)(wq_r5[kq_7] * 48 + 8 + 8 * wq_b5[kq_7])), "l"(tokp + ((unsigned long long)rowa_5 * 24 + (unsigned long long)wq_w5[kq_7])));
                                        }
                                    }
                                }
                                asm volatile("cp.async.commit_group;");
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
                        float _sub_0;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(_tmem_load_4[n_1]), "f"(comp[n_1]));
                        float _add_0;
                        asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(acc[n_1]), "f"(_sub_0));
                        float _sub_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(_add_0), "f"(acc[n_1]));
                        float _sub_0_1;
                        asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_0_1) : "f"(_sub_1), "f"(_sub_0));
                        comp[n_1] = _sub_0_1;
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
                            float _sub_1_1;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_1_1) : "f"(_tmem_load_5[n_2]), "f"(comp[n_2]));
                            float _add_1;
                            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(acc[n_2]), "f"(_sub_1_1));
                            float _sub_3;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(_add_1), "f"(acc[n_2]));
                            float _sub_2;
                            asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(_sub_3), "f"(_sub_1_1));
                            comp[n_2] = _sub_2;
                            acc[n_2] = _add_1;
                        }
                        grp_ep = grp_ep + 1;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(acc_empty_addr + (acc_ep) * 8);
                    }
                    acc_ep += 1;
                    if (acc_ep == 2) { acc_ep = 0; _phase_acc_full ^= 1; _phase_dfn_full ^= 1; }
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
                    if (ep_stage == 2) { ep_stage = 0; _phase_stage_full_x_1 ^= 1; _phase_stage_full ^= 1; }
                }
                unsigned long long kcol = (unsigned long long)(i_str * 4096 + s_2 * 64 + dl);
                unsigned long long pbase = (unsigned long long)p_3 * 393216;
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
