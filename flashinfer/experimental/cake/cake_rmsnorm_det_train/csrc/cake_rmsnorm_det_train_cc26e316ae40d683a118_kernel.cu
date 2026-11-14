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

#define LAUNCH_MIN_BLOCKS 3

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



extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_cc26e316ae40d683a118(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dots = reinterpret_cast<float*>(smem_raw + 1024);
    const int dots_addr = smem + 1024;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1152);
    const int ring_addr = smem + 1152;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            mbarrier_init(smem + 40, 128);
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int t = tid;
            int ring_at = (int)ring_addr;
            unsigned int ww[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)((0 + t) * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)((128 + t) * 8));
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
            float acc[16];
            unsigned int gw[8];
            unsigned int xw[8];
            int k = 0;
            int slot = (int)bid;
            while (slot < n_chunks) {
                acc[0] = 0.0f;
                acc[1] = 0.0f;
                acc[2] = 0.0f;
                acc[3] = 0.0f;
                acc[4] = 0.0f;
                acc[5] = 0.0f;
                acc[6] = 0.0f;
                acc[7] = 0.0f;
                acc[8] = 0.0f;
                acc[9] = 0.0f;
                acc[10] = 0.0f;
                acc[11] = 0.0f;
                acc[12] = 0.0f;
                acc[13] = 0.0f;
                acc[14] = 0.0f;
                acc[15] = 0.0f;
                int row = slot;
                while (row < T) {
                    float rq = r[row];
                    int _stage_full = (int)((unsigned int)k % 4);
                    int _parity_full = (int)((unsigned int)k / 4 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + k % 4 * 8192;
                    __int128_t _smem_raw_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(sbase + (0 + t) * 8 * 2));
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                    __int128_t _smem_raw_3;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(sbase + 4096 + (0 + t) * 8 * 2));
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(sbase + (128 + t) * 8 * 2));
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(sbase + 4096 + (128 + t) * 8 * 2));
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    float dq = 0.0f;
                    float ga_0 = __uint_as_float(gw[0] << 16);
                    float gb_0 = __uint_as_float(gw[0] & 4294901760u);
                    float xa_0 = __uint_as_float(xw[0] << 16);
                    float xb_0 = __uint_as_float(xw[0] & 4294901760u);
                    dq = __fmaf_rn(ga_0 * __uint_as_float(ww[0] << 16), xa_0, dq);
                    dq = __fmaf_rn(gb_0 * __uint_as_float(ww[0] & 4294901760u), xb_0, dq);
                    acc[0] = __fmaf_rn(ga_0 * xa_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(gb_0 * xb_0, rq, acc[1]);
                    float ga_1 = __uint_as_float(gw[1] << 16);
                    float gb_1 = __uint_as_float(gw[1] & 4294901760u);
                    float xa_1 = __uint_as_float(xw[1] << 16);
                    float xb_1 = __uint_as_float(xw[1] & 4294901760u);
                    dq = __fmaf_rn(ga_1 * __uint_as_float(ww[1] << 16), xa_1, dq);
                    dq = __fmaf_rn(gb_1 * __uint_as_float(ww[1] & 4294901760u), xb_1, dq);
                    acc[2] = __fmaf_rn(ga_1 * xa_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(gb_1 * xb_1, rq, acc[3]);
                    float ga_2 = __uint_as_float(gw[2] << 16);
                    float gb_2 = __uint_as_float(gw[2] & 4294901760u);
                    float xa_2 = __uint_as_float(xw[2] << 16);
                    float xb_2 = __uint_as_float(xw[2] & 4294901760u);
                    dq = __fmaf_rn(ga_2 * __uint_as_float(ww[2] << 16), xa_2, dq);
                    dq = __fmaf_rn(gb_2 * __uint_as_float(ww[2] & 4294901760u), xb_2, dq);
                    acc[4] = __fmaf_rn(ga_2 * xa_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(gb_2 * xb_2, rq, acc[5]);
                    float ga_3 = __uint_as_float(gw[3] << 16);
                    float gb_3 = __uint_as_float(gw[3] & 4294901760u);
                    float xa_3 = __uint_as_float(xw[3] << 16);
                    float xb_3 = __uint_as_float(xw[3] & 4294901760u);
                    dq = __fmaf_rn(ga_3 * __uint_as_float(ww[3] << 16), xa_3, dq);
                    dq = __fmaf_rn(gb_3 * __uint_as_float(ww[3] & 4294901760u), xb_3, dq);
                    acc[6] = __fmaf_rn(ga_3 * xa_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(gb_3 * xb_3, rq, acc[7]);
                    float ga_4 = __uint_as_float(gw[4] << 16);
                    float gb_4 = __uint_as_float(gw[4] & 4294901760u);
                    float xa_4 = __uint_as_float(xw[4] << 16);
                    float xb_4 = __uint_as_float(xw[4] & 4294901760u);
                    dq = __fmaf_rn(ga_4 * __uint_as_float(ww[4] << 16), xa_4, dq);
                    dq = __fmaf_rn(gb_4 * __uint_as_float(ww[4] & 4294901760u), xb_4, dq);
                    acc[8] = __fmaf_rn(ga_4 * xa_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(gb_4 * xb_4, rq, acc[9]);
                    float ga_5 = __uint_as_float(gw[5] << 16);
                    float gb_5 = __uint_as_float(gw[5] & 4294901760u);
                    float xa_5 = __uint_as_float(xw[5] << 16);
                    float xb_5 = __uint_as_float(xw[5] & 4294901760u);
                    dq = __fmaf_rn(ga_5 * __uint_as_float(ww[5] << 16), xa_5, dq);
                    dq = __fmaf_rn(gb_5 * __uint_as_float(ww[5] & 4294901760u), xb_5, dq);
                    acc[10] = __fmaf_rn(ga_5 * xa_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(gb_5 * xb_5, rq, acc[11]);
                    float ga_6 = __uint_as_float(gw[6] << 16);
                    float gb_6 = __uint_as_float(gw[6] & 4294901760u);
                    float xa_6 = __uint_as_float(xw[6] << 16);
                    float xb_6 = __uint_as_float(xw[6] & 4294901760u);
                    dq = __fmaf_rn(ga_6 * __uint_as_float(ww[6] << 16), xa_6, dq);
                    dq = __fmaf_rn(gb_6 * __uint_as_float(ww[6] & 4294901760u), xb_6, dq);
                    acc[12] = __fmaf_rn(ga_6 * xa_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(gb_6 * xb_6, rq, acc[13]);
                    float ga_7 = __uint_as_float(gw[7] << 16);
                    float gb_7 = __uint_as_float(gw[7] & 4294901760u);
                    float xa_7 = __uint_as_float(xw[7] << 16);
                    float xb_7 = __uint_as_float(xw[7] & 4294901760u);
                    dq = __fmaf_rn(ga_7 * __uint_as_float(ww[7] << 16), xa_7, dq);
                    dq = __fmaf_rn(gb_7 * __uint_as_float(ww[7] & 4294901760u), xb_7, dq);
                    acc[14] = __fmaf_rn(ga_7 * xa_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(gb_7 * xb_7, rq, acc[15]);
                    float dsum = dq;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsum += __shfl_xor_sync(0xFFFFFFFF, dsum, offset);
                    int par = (k & 1) * 4;
                    if (lane == 0) {
                        dots[par + warp] = dsum;
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    float tot = dots[par];
                    tot = tot + dots[par + 1];
                    tot = tot + dots[par + 1 + 1];
                    tot = tot + dots[par + 2 + 1];
                    float cq = tot * 0.00048828125f;
                    float kx = rq * rq * rq * cq;
                    float nkx = -kx;
                    unsigned long long dx_base = (unsigned long long)row * 2048ull;
                    __int128_t _smem_raw_6;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(sbase + (0 + t) * 8 * 2));
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(sbase + 4096 + (0 + t) * 8 * 2));
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    out_0[0] = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
                    out_0[1] = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    out_0[2] = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
                    out_0[3] = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    out_0[4] = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
                    out_0[5] = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    out_0[6] = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(ww[3] << 16)));
                    out_0[7] = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((0 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(sbase + (128 + t) * 8 * 2));
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    __int128_t _smem_raw_9;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(sbase + 4096 + (128 + t) * 8 * 2));
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    out_1[0] = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
                    out_1[1] = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    out_1[2] = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
                    out_1[3] = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    out_1[4] = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
                    out_1[5] = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    out_1[6] = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(ww[7] << 16)));
                    out_1[7] = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(ww[7] & 4294901760u)));
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((128 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 4)) * 8);
                    row = row + n_chunks;
                    k += 1;
                }
                unsigned long long pbase = (unsigned long long)slot * 2048ull;
                {
                    unsigned _stv8_10_0 = __float_as_uint(acc[0 + 0]);
                    unsigned _stv8_10_1 = __float_as_uint(acc[0 + 1]);
                    unsigned _stv8_10_2 = __float_as_uint(acc[0 + 2]);
                    unsigned _stv8_10_3 = __float_as_uint(acc[0 + 3]);
                    unsigned _stv8_10_4 = __float_as_uint(acc[0 + 4]);
                    unsigned _stv8_10_5 = __float_as_uint(acc[0 + 5]);
                    unsigned _stv8_10_6 = __float_as_uint(acc[0 + 6]);
                    unsigned _stv8_10_7 = __float_as_uint(acc[0 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((0 + t) * 8)))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                }
                {
                    unsigned _stv8_11_0 = __float_as_uint(acc[8 + 0]);
                    unsigned _stv8_11_1 = __float_as_uint(acc[8 + 1]);
                    unsigned _stv8_11_2 = __float_as_uint(acc[8 + 2]);
                    unsigned _stv8_11_3 = __float_as_uint(acc[8 + 3]);
                    unsigned _stv8_11_4 = __float_as_uint(acc[8 + 4]);
                    unsigned _stv8_11_5 = __float_as_uint(acc[8 + 5]);
                    unsigned _stv8_11_6 = __float_as_uint(acc[8 + 6]);
                    unsigned _stv8_11_7 = __float_as_uint(acc[8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((128 + t) * 8)))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                }
                slot = slot + (int)num_bids;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 4) {
        { // prod_main
            if (elect_sync()) {
                int p_ring = (int)ring_addr;
                int p_slot = (int)bid;
                int p_row = p_slot;
                #pragma unroll 1
                for (int pk = 0; pk < T; pk++) {
                    if (p_slot >= n_chunks) {
                        break;
                    }
                    int _stage_empty = (int)((unsigned int)pk % 4);
                    int _parity_empty = (int)((unsigned int)pk / 4 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 8192);
                    int sb = p_ring + pk % 4 * 8192;
                    long long src = (long long)p_row;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sb), "l"(reinterpret_cast<const uint8_t*>(g + src * g_stride_p0)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                           "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sb + 4096), "l"(reinterpret_cast<const uint8_t*>(x + src * x_stride_p0)), "r"((unsigned)(4096)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                           "l"(0x12F0000000000000ULL) : "memory");
                    p_row = p_row + n_chunks;
                    if (p_row >= T) {
                        p_slot = p_slot + (int)num_bids;
                        p_row = p_slot;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
