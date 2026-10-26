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
#define TMEM_NCOLS 256
#define TMEM_ACC1_OFFSET 0
#define TMEM_ACC2_OFFSET 128
#define NUM_ONE_STAGES 1
#define SMEM_DO_PANEL_OFF 1024
#define SMEM_DO_PANEL_STAGE_BYTES 16384
#define SMEM_DO_PANEL_STRIDE 16384
#define SMEM_VNEW_PANEL_OFF 33792
#define SMEM_VNEW_PANEL_STAGE_BYTES 16384
#define SMEM_VNEW_PANEL_STRIDE 16384
#define SMEM_AQK_PANEL_OFF 66560
#define SMEM_AQK_PANEL_STAGE_BYTES 8192
#define SMEM_AQK_PANEL_STRIDE 8192
#define SMEM_DO_KMAJ_OFF 1024
#define SMEM_DO_KMAJ_STAGE_BYTES 32768
#define SMEM_DO_KMAJ_STRIDE 32768
#define SMEM_VNEW_KMAJ_OFF 33792
#define SMEM_VNEW_KMAJ_STAGE_BYTES 32768
#define SMEM_VNEW_KMAJ_STRIDE 32768
#define SMEM_DO_MN_OFF 1024
#define SMEM_DO_MN_STAGE_BYTES 32768
#define SMEM_DO_MN_STRIDE 32768
#define SMEM_AQK_MN_OFF 66560
#define SMEM_AQK_MN_STAGE_BYTES 32768
#define SMEM_AQK_MN_STRIDE 32768
#define SMEM_TOTAL 99328
#define THREADS 192
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_train_f227dd21f1f3cb464b29(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap vnew_tma, const __grid_constant__ CUtensorMap aqk_tma, float* __restrict__ dAqk, __nv_bfloat16* __restrict__ dv1, int num_heads, float scale)
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
    #define tiles_full_addr (mbar_base + 0)
    #define zero_done_addr (mbar_base + 8)
    #define acc_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_panel_addr = smem + 1024;
    __nv_bfloat16* vnew_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vnew_panel_addr = smem + 33792;
    __nv_bfloat16* aqk_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int aqk_panel_addr = smem + 66560;
    __nv_bfloat16* do_kmaj = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_kmaj_addr = smem + 1024;
    __nv_bfloat16* vnew_kmaj = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int vnew_kmaj_addr = smem + 33792;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int do_mn_addr = smem + 1024;
    __nv_bfloat16* aqk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int aqk_mn_addr = smem + 66560;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // tiles_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // zero_done: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 24);
    if (warp == 0) {
        int _tmem_hold = smem + 24;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc1 = taddr;
    const int tmem_acc2 = taddr + 128;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int pair = blockIdx.x;
            int head = blockIdx.y;
            int row0 = pair * 128;
            const int epi_warp = warp;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            #pragma unroll
            for (int blk = 1; blk < 3; blk++) {
                int base = aqk_panel_addr + (unsigned int)(blk * 8192);
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base + (j * 128 + epi_tid) * 16), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            unsigned int _phase_acc_full_0 = 0;
            mbarrier_wait(acc_full_addr, _phase_acc_full_0);
            _phase_acc_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            const int row_base = epi_warp * 32;
            int row = row_base + lane;
            int chunk = row / 64;
            int t_local = row - chunk * 64;
            int grow = row0 + row;
            int dA_base = (grow * num_heads + head) * 64;
            #pragma unroll
            for (int seg = 0; seg < 4; seg++) {
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], taddr + (unsigned int)(row_base << 16) + (unsigned int)(chunk * 64) + (unsigned int)(seg * 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    float out8[8];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        int s_local = seg * 16 + half * 8 + j_1;
                        float val = _tmem_load_0[half * 8 + j_1] * scale;
                        if (s_local > t_local) {
                            val = 0.0f;
                        }
                        out8[j_1] = val;
                    }
                    {
                        unsigned _stv8_0_0 = __float_as_uint(out8[0 + 0]);
                        unsigned _stv8_0_1 = __float_as_uint(out8[0 + 1]);
                        unsigned _stv8_0_2 = __float_as_uint(out8[0 + 2]);
                        unsigned _stv8_0_3 = __float_as_uint(out8[0 + 3]);
                        unsigned _stv8_0_4 = __float_as_uint(out8[0 + 4]);
                        unsigned _stv8_0_5 = __float_as_uint(out8[0 + 5]);
                        unsigned _stv8_0_6 = __float_as_uint(out8[0 + 6]);
                        unsigned _stv8_0_7 = __float_as_uint(out8[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAqk + (dA_base + seg * 16 + half * 8) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                    }
                }
            }
            long long dv_base = ((long long)grow * (long long)num_heads + (long long)head) * 128;
            #pragma unroll
            for (int seg_1 = 0; seg_1 < 8; seg_1++) {
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)(row_base << 16) + 128 + (unsigned int)(seg_1 * 16));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                unsigned int _packed_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 2], _tmem_load_1[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 4], _tmem_load_1[_lp*2+1 + 4]));
                    _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 6], _tmem_load_1[_lp*2+1 + 6]));
                    _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 8], _tmem_load_1[_lp*2+1 + 8]));
                    _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 10], _tmem_load_1[_lp*2+1 + 10]));
                    _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 12], _tmem_load_1[_lp*2+1 + 12]));
                    _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 14], _tmem_load_1[_lp*2+1 + 14]));
                    _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                }
                {
                    const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(_packed_0);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(dv1 + (dv_base + (long long)(seg_1 * 16)))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int pair_1 = blockIdx.x;
            int head_1 = blockIdx.y;
            int row0_1 = pair_1 * 128;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(tiles_full_addr, 81920);
                #pragma unroll
                for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                    tma_3d_gmem2smem(do_panel_addr + (unsigned int)(seg_2 * 16384), (&do_tma), seg_2 * 64, head_1, row0_1, tiles_full_addr);
                    tma_3d_gmem2smem(vnew_panel_addr + (unsigned int)(seg_2 * 16384), (&vnew_tma), seg_2 * 64, head_1, row0_1, tiles_full_addr);
                }
                tma_3d_gmem2smem(aqk_panel_addr, (&aqk_tma), 0, head_1, row0_1, tiles_full_addr);
                tma_3d_gmem2smem(aqk_panel_addr + 24576, (&aqk_tma), 0, head_1, row0_1 + 64, tiles_full_addr);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int _phase_tiles_full_0 = 0;
            unsigned int _phase_zero_done_0 = 0;
            if (elect_sync()) {
                mbarrier_wait(tiles_full_addr, _phase_tiles_full_0);
                _phase_tiles_full_0 ^= 1;
                mbarrier_wait(zero_done_addr, _phase_zero_done_0);
                _phase_zero_done_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_0 = ((do_kmaj_addr) >> 4) & 0x7FFF;
                int _smem_desc_lo_1 = ((vnew_kmaj_addr) >> 4) & 0x7FFF;
#else
                int _smem_desc_lo_0 = ((do_kmaj_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_1 = ((vnew_kmaj_addr) >> 4) & 0x3FFF;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc1), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_2 = (((aqk_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
                int _smem_desc_lo_3 = (((do_mn_addr) >> 4) & 0x7FFF) | 0x4000000;
#else
                int _smem_desc_lo_2 = (((aqk_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
                int _smem_desc_lo_3 = (((do_mn_addr) >> 4) & 0x3FFF) | 0x4000000;
#endif
                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136414352;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc2), "r"(0) : "memory");
                tcgen05_commit(acc_full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
