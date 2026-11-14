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

__global__ __launch_bounds__(416, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_7c216db7f9e0c6fd9c4d(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial));
    __builtin_assume(__isGlobal(partial));

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
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    double* sums = reinterpret_cast<double*>(smem_raw + 1184);
    const int sums_addr = smem + 1184;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1184);
    const int dwbuf_addr = smem + 1184;
    float* pairs = reinterpret_cast<float*>(smem_raw + 1184);
    const int pairs_addr = smem + 1184;
    int* meta = reinterpret_cast<int*>(smem_raw + 1152);
    const int meta_addr = smem + 1152;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 50432);
    const int ring_addr = smem + 50432;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=384
            mbarrier_init(smem + 32, 384);
            mbarrier_init(smem + 40, 384);
            mbarrier_init(smem + 48, 384);
            mbarrier_init(smem + 56, 384);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 11) {
        { // compute_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int nxt = 0;
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            long long g_stride = g_stride_p0;
            long long x_stride = x_stride_p0;
            long long gh_stride = g_h_stride_p0;
            int rg = warp / 12;
            int tir = tid - rg * 384;
            unsigned int ww[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)((384 + tir) * 8));
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
            float acc[16];
            float comp[1];
            float hr[1];
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
            int c = 0;
            int first_c = 0;
            sums[rg * 6144 + tir] = 0.0;
            sums[rg * 6144 + 384 + tir] = 0.0;
            sums[rg * 6144 + 768 + tir] = 0.0;
            sums[rg * 6144 + 1152 + tir] = 0.0;
            sums[rg * 6144 + 1536 + tir] = 0.0;
            sums[rg * 6144 + 1920 + tir] = 0.0;
            sums[rg * 6144 + 2304 + tir] = 0.0;
            sums[rg * 6144 + 2688 + tir] = 0.0;
            sums[rg * 6144 + 3072 + tir] = 0.0;
            sums[rg * 6144 + 3456 + tir] = 0.0;
            sums[rg * 6144 + 3840 + tir] = 0.0;
            sums[rg * 6144 + 4224 + tir] = 0.0;
            sums[rg * 6144 + 4608 + tir] = 0.0;
            sums[rg * 6144 + 4992 + tir] = 0.0;
            sums[rg * 6144 + 5376 + tir] = 0.0;
            sums[rg * 6144 + 5760 + tir] = 0.0;
            int ring_at = (int)ring_addr;
            unsigned int gw[8];
            unsigned int xw[8];
            unsigned int hw[8];
            float rr[1];
            float dot[1];
            #pragma unroll 1
            for (int k = 0; k < T + n_chunks + 1; k++) {
                int _stage_full = (int)((unsigned int)k % 4);
                int _parity_full = (int)((unsigned int)k / 4 % 2);
                mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                int row0 = meta[2 * (k % 4)];
                int nval = meta[2 * (k % 4) + 1];
                if (nval < 1) {
                    c = row0;
                    first_c = -1 - nval;
                    break;
                }
                int rbase = ring_at + k % 4 * 36864 + rg * 12288;
                __int128_t _smem_raw_2;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(rbase + tir * 8 * 2));
                gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
                gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
                gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
                gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(rbase + 12288 + tir * 8 * 2));
                xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(rbase + 24576 + tir * 8 * 2));
                hw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                hw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                hw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                hw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                __int128_t _smem_raw_5;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(rbase + (384 + tir) * 8 * 2));
                gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                __int128_t _smem_raw_6;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
                xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                __int128_t _smem_raw_7;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(rbase + 24576 + (384 + tir) * 8 * 2));
                hw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                hw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                hw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                hw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 4)) * 8);
                rr[0] = 0.0f;
                if (nval > rg) {
                    rr[0] = r[row0 + rg];
                }
                float dq = 0.0f;
                dq = __fmaf_rn(__uint_as_float(gw[0] << 16) * __uint_as_float(ww[0] << 16), __uint_as_float(xw[0] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[0] & 4294901760u) * __uint_as_float(ww[0] & 4294901760u), __uint_as_float(xw[0] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] << 16) * __uint_as_float(ww[1] << 16), __uint_as_float(xw[1] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[1] & 4294901760u) * __uint_as_float(ww[1] & 4294901760u), __uint_as_float(xw[1] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] << 16) * __uint_as_float(ww[2] << 16), __uint_as_float(xw[2] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[2] & 4294901760u) * __uint_as_float(ww[2] & 4294901760u), __uint_as_float(xw[2] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] << 16) * __uint_as_float(ww[3] << 16), __uint_as_float(xw[3] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[3] & 4294901760u) * __uint_as_float(ww[3] & 4294901760u), __uint_as_float(xw[3] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] << 16) * __uint_as_float(ww[4] << 16), __uint_as_float(xw[4] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[4] & 4294901760u) * __uint_as_float(ww[4] & 4294901760u), __uint_as_float(xw[4] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] << 16) * __uint_as_float(ww[5] << 16), __uint_as_float(xw[5] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[5] & 4294901760u) * __uint_as_float(ww[5] & 4294901760u), __uint_as_float(xw[5] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] << 16) * __uint_as_float(ww[6] << 16), __uint_as_float(xw[6] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[6] & 4294901760u) * __uint_as_float(ww[6] & 4294901760u), __uint_as_float(xw[6] & 4294901760u), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] << 16) * __uint_as_float(ww[7] << 16), __uint_as_float(xw[7] << 16), dq);
                dq = __fmaf_rn(__uint_as_float(gw[7] & 4294901760u) * __uint_as_float(ww[7] & 4294901760u), __uint_as_float(xw[7] & 4294901760u), dq);
                float _warp_reduce_8 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
                dot[0] = _warp_reduce_8;
                if (lane == 0) {
                    dots[(k & 1) * 12 + warp] = dot[0];
                }
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                float tot = 0.0f;
                tot += dots[(k & 1) * 12 + rg * 12];
                tot += dots[(k & 1) * 12 + rg * 12 + 1];
                tot += dots[(k & 1) * 12 + rg * 12 + 2];
                tot += dots[(k & 1) * 12 + rg * 12 + 3];
                tot += dots[(k & 1) * 12 + rg * 12 + 4];
                tot += dots[(k & 1) * 12 + rg * 12 + 5];
                tot += dots[(k & 1) * 12 + rg * 12 + 6];
                tot += dots[(k & 1) * 12 + rg * 12 + 7];
                tot += dots[(k & 1) * 12 + rg * 12 + 8];
                tot += dots[(k & 1) * 12 + rg * 12 + 9];
                tot += dots[(k & 1) * 12 + rg * 12 + 10];
                tot += dots[(k & 1) * 12 + rg * 12 + 11];
                dot[0] = tot;
                float rq = rr[0];
                float cq = __fdiv_rn(dot[0], 6144.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row0 + rg < row0 + nval) {
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    acc[0] = __fmaf_rn(g0_0 * x0_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(g1_0 * x1_0, rq, acc[1]);
                    float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
                    float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                    out_0[0] = y0_0 + __uint_as_float(hw[0] << 16);
                    out_0[1] = y1_0 + __uint_as_float(hw[0] & 4294901760u);
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    acc[2] = __fmaf_rn(g0_1 * x0_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(g1_1 * x1_1, rq, acc[3]);
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                    out_0[2] = y0_1 + __uint_as_float(hw[1] << 16);
                    out_0[3] = y1_1 + __uint_as_float(hw[1] & 4294901760u);
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    acc[4] = __fmaf_rn(g0_2 * x0_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(g1_2 * x1_2, rq, acc[5]);
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                    out_0[4] = y0_2 + __uint_as_float(hw[2] << 16);
                    out_0[5] = y1_2 + __uint_as_float(hw[2] & 4294901760u);
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    acc[6] = __fmaf_rn(g0_3 * x0_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(g1_3 * x1_3, rq, acc[7]);
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(ww[3] << 16)));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                    out_0[6] = y0_3 + __uint_as_float(hw[3] << 16);
                    out_0[7] = y1_3 + __uint_as_float(hw[3] & 4294901760u);
                    unsigned int packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_0[_lp*2 + 0], out_0[_lp*2+1 + 0]));
                        packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
                    float out_1[8];
                    float g0_4 = __uint_as_float(gw[4] << 16);
                    float g1_4 = __uint_as_float(gw[4] & 4294901760u);
                    float x0_4 = __uint_as_float(xw[4] << 16);
                    float x1_4 = __uint_as_float(xw[4] & 4294901760u);
                    acc[8] = __fmaf_rn(g0_4 * x0_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(g1_4 * x1_4, rq, acc[9]);
                    float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
                    float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
                    out_1[0] = y0_4 + __uint_as_float(hw[4] << 16);
                    out_1[1] = y1_4 + __uint_as_float(hw[4] & 4294901760u);
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    acc[10] = __fmaf_rn(g0_5 * x0_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(g1_5 * x1_5, rq, acc[11]);
                    float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
                    float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
                    out_1[2] = y0_5 + __uint_as_float(hw[5] << 16);
                    out_1[3] = y1_5 + __uint_as_float(hw[5] & 4294901760u);
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    acc[12] = __fmaf_rn(g0_6 * x0_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(g1_6 * x1_6, rq, acc[13]);
                    float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
                    float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
                    out_1[4] = y0_6 + __uint_as_float(hw[6] << 16);
                    out_1[5] = y1_6 + __uint_as_float(hw[6] & 4294901760u);
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    acc[14] = __fmaf_rn(g0_7 * x0_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(g1_7 * x1_7, rq, acc[15]);
                    float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(ww[7] << 16)));
                    float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(ww[7] & 4294901760u)));
                    out_1[6] = y0_7 + __uint_as_float(hw[7] << 16);
                    out_1[7] = y1_7 + __uint_as_float(hw[7] & 4294901760u);
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
                if ((k + 1 & 7) == 0) {
                    float hi_0 = pairs[2 * (rg * 6144 + tir)];
                    float s_0 = hi_0 + acc[0];
                    float bb_0 = s_0 - hi_0;
                    float err_0 = hi_0 - (s_0 - bb_0) + (acc[0] - bb_0);
                    pairs[2 * (rg * 6144 + tir) + 1] = pairs[2 * (rg * 6144 + tir) + 1] + err_0;
                    pairs[2 * (rg * 6144 + tir)] = s_0;
                    acc[0] = 0.0f;
                    float hi_1 = pairs[2 * (rg * 6144 + 384 + tir)];
                    float s_1 = hi_1 + acc[1];
                    float bb_1 = s_1 - hi_1;
                    float err_1 = hi_1 - (s_1 - bb_1) + (acc[1] - bb_1);
                    pairs[2 * (rg * 6144 + 384 + tir) + 1] = pairs[2 * (rg * 6144 + 384 + tir) + 1] + err_1;
                    pairs[2 * (rg * 6144 + 384 + tir)] = s_1;
                    acc[1] = 0.0f;
                    float hi_2 = pairs[2 * (rg * 6144 + 768 + tir)];
                    float s_2 = hi_2 + acc[2];
                    float bb_2 = s_2 - hi_2;
                    float err_2 = hi_2 - (s_2 - bb_2) + (acc[2] - bb_2);
                    pairs[2 * (rg * 6144 + 768 + tir) + 1] = pairs[2 * (rg * 6144 + 768 + tir) + 1] + err_2;
                    pairs[2 * (rg * 6144 + 768 + tir)] = s_2;
                    acc[2] = 0.0f;
                    float hi_3 = pairs[2 * (rg * 6144 + 1152 + tir)];
                    float s_3 = hi_3 + acc[3];
                    float bb_3 = s_3 - hi_3;
                    float err_3 = hi_3 - (s_3 - bb_3) + (acc[3] - bb_3);
                    pairs[2 * (rg * 6144 + 1152 + tir) + 1] = pairs[2 * (rg * 6144 + 1152 + tir) + 1] + err_3;
                    pairs[2 * (rg * 6144 + 1152 + tir)] = s_3;
                    acc[3] = 0.0f;
                    float hi_4 = pairs[2 * (rg * 6144 + 1536 + tir)];
                    float s_4 = hi_4 + acc[4];
                    float bb_4 = s_4 - hi_4;
                    float err_4 = hi_4 - (s_4 - bb_4) + (acc[4] - bb_4);
                    pairs[2 * (rg * 6144 + 1536 + tir) + 1] = pairs[2 * (rg * 6144 + 1536 + tir) + 1] + err_4;
                    pairs[2 * (rg * 6144 + 1536 + tir)] = s_4;
                    acc[4] = 0.0f;
                    float hi_5 = pairs[2 * (rg * 6144 + 1920 + tir)];
                    float s_5 = hi_5 + acc[5];
                    float bb_5 = s_5 - hi_5;
                    float err_5 = hi_5 - (s_5 - bb_5) + (acc[5] - bb_5);
                    pairs[2 * (rg * 6144 + 1920 + tir) + 1] = pairs[2 * (rg * 6144 + 1920 + tir) + 1] + err_5;
                    pairs[2 * (rg * 6144 + 1920 + tir)] = s_5;
                    acc[5] = 0.0f;
                    float hi_6 = pairs[2 * (rg * 6144 + 2304 + tir)];
                    float s_6 = hi_6 + acc[6];
                    float bb_6 = s_6 - hi_6;
                    float err_6 = hi_6 - (s_6 - bb_6) + (acc[6] - bb_6);
                    pairs[2 * (rg * 6144 + 2304 + tir) + 1] = pairs[2 * (rg * 6144 + 2304 + tir) + 1] + err_6;
                    pairs[2 * (rg * 6144 + 2304 + tir)] = s_6;
                    acc[6] = 0.0f;
                    float hi_7 = pairs[2 * (rg * 6144 + 2688 + tir)];
                    float s_7 = hi_7 + acc[7];
                    float bb_7 = s_7 - hi_7;
                    float err_7 = hi_7 - (s_7 - bb_7) + (acc[7] - bb_7);
                    pairs[2 * (rg * 6144 + 2688 + tir) + 1] = pairs[2 * (rg * 6144 + 2688 + tir) + 1] + err_7;
                    pairs[2 * (rg * 6144 + 2688 + tir)] = s_7;
                    acc[7] = 0.0f;
                    float hi_8 = pairs[2 * (rg * 6144 + 3072 + tir)];
                    float s_8 = hi_8 + acc[8];
                    float bb_8 = s_8 - hi_8;
                    float err_8 = hi_8 - (s_8 - bb_8) + (acc[8] - bb_8);
                    pairs[2 * (rg * 6144 + 3072 + tir) + 1] = pairs[2 * (rg * 6144 + 3072 + tir) + 1] + err_8;
                    pairs[2 * (rg * 6144 + 3072 + tir)] = s_8;
                    acc[8] = 0.0f;
                    float hi_9 = pairs[2 * (rg * 6144 + 3456 + tir)];
                    float s_9 = hi_9 + acc[9];
                    float bb_9 = s_9 - hi_9;
                    float err_9 = hi_9 - (s_9 - bb_9) + (acc[9] - bb_9);
                    pairs[2 * (rg * 6144 + 3456 + tir) + 1] = pairs[2 * (rg * 6144 + 3456 + tir) + 1] + err_9;
                    pairs[2 * (rg * 6144 + 3456 + tir)] = s_9;
                    acc[9] = 0.0f;
                    float hi_10 = pairs[2 * (rg * 6144 + 3840 + tir)];
                    float s_10 = hi_10 + acc[10];
                    float bb_10 = s_10 - hi_10;
                    float err_10 = hi_10 - (s_10 - bb_10) + (acc[10] - bb_10);
                    pairs[2 * (rg * 6144 + 3840 + tir) + 1] = pairs[2 * (rg * 6144 + 3840 + tir) + 1] + err_10;
                    pairs[2 * (rg * 6144 + 3840 + tir)] = s_10;
                    acc[10] = 0.0f;
                    float hi_11 = pairs[2 * (rg * 6144 + 4224 + tir)];
                    float s_11 = hi_11 + acc[11];
                    float bb_11 = s_11 - hi_11;
                    float err_11 = hi_11 - (s_11 - bb_11) + (acc[11] - bb_11);
                    pairs[2 * (rg * 6144 + 4224 + tir) + 1] = pairs[2 * (rg * 6144 + 4224 + tir) + 1] + err_11;
                    pairs[2 * (rg * 6144 + 4224 + tir)] = s_11;
                    acc[11] = 0.0f;
                    float hi_12 = pairs[2 * (rg * 6144 + 4608 + tir)];
                    float s_12 = hi_12 + acc[12];
                    float bb_12 = s_12 - hi_12;
                    float err_12 = hi_12 - (s_12 - bb_12) + (acc[12] - bb_12);
                    pairs[2 * (rg * 6144 + 4608 + tir) + 1] = pairs[2 * (rg * 6144 + 4608 + tir) + 1] + err_12;
                    pairs[2 * (rg * 6144 + 4608 + tir)] = s_12;
                    acc[12] = 0.0f;
                    float hi_13 = pairs[2 * (rg * 6144 + 4992 + tir)];
                    float s_13 = hi_13 + acc[13];
                    float bb_13 = s_13 - hi_13;
                    float err_13 = hi_13 - (s_13 - bb_13) + (acc[13] - bb_13);
                    pairs[2 * (rg * 6144 + 4992 + tir) + 1] = pairs[2 * (rg * 6144 + 4992 + tir) + 1] + err_13;
                    pairs[2 * (rg * 6144 + 4992 + tir)] = s_13;
                    acc[13] = 0.0f;
                    float hi_14 = pairs[2 * (rg * 6144 + 5376 + tir)];
                    float s_14 = hi_14 + acc[14];
                    float bb_14 = s_14 - hi_14;
                    float err_14 = hi_14 - (s_14 - bb_14) + (acc[14] - bb_14);
                    pairs[2 * (rg * 6144 + 5376 + tir) + 1] = pairs[2 * (rg * 6144 + 5376 + tir) + 1] + err_14;
                    pairs[2 * (rg * 6144 + 5376 + tir)] = s_14;
                    acc[14] = 0.0f;
                    float hi_15 = pairs[2 * (rg * 6144 + 5760 + tir)];
                    float s_15 = hi_15 + acc[15];
                    float bb_15 = s_15 - hi_15;
                    float err_15 = hi_15 - (s_15 - bb_15) + (acc[15] - bb_15);
                    pairs[2 * (rg * 6144 + 5760 + tir) + 1] = pairs[2 * (rg * 6144 + 5760 + tir) + 1] + err_15;
                    pairs[2 * (rg * 6144 + 5760 + tir)] = s_15;
                    acc[15] = 0.0f;
                }
            }
            float hi_16 = pairs[2 * (rg * 6144 + tir)];
            float s_16 = hi_16 + acc[0];
            float bb_16 = s_16 - hi_16;
            float err_16 = hi_16 - (s_16 - bb_16) + (acc[0] - bb_16);
            pairs[2 * (rg * 6144 + tir) + 1] = pairs[2 * (rg * 6144 + tir) + 1] + err_16;
            pairs[2 * (rg * 6144 + tir)] = s_16;
            acc[0] = 0.0f;
            float hi_17 = pairs[2 * (rg * 6144 + 384 + tir)];
            float s_17 = hi_17 + acc[1];
            float bb_17 = s_17 - hi_17;
            float err_17 = hi_17 - (s_17 - bb_17) + (acc[1] - bb_17);
            pairs[2 * (rg * 6144 + 384 + tir) + 1] = pairs[2 * (rg * 6144 + 384 + tir) + 1] + err_17;
            pairs[2 * (rg * 6144 + 384 + tir)] = s_17;
            acc[1] = 0.0f;
            float hi_18 = pairs[2 * (rg * 6144 + 768 + tir)];
            float s_18 = hi_18 + acc[2];
            float bb_18 = s_18 - hi_18;
            float err_18 = hi_18 - (s_18 - bb_18) + (acc[2] - bb_18);
            pairs[2 * (rg * 6144 + 768 + tir) + 1] = pairs[2 * (rg * 6144 + 768 + tir) + 1] + err_18;
            pairs[2 * (rg * 6144 + 768 + tir)] = s_18;
            acc[2] = 0.0f;
            float hi_19 = pairs[2 * (rg * 6144 + 1152 + tir)];
            float s_19 = hi_19 + acc[3];
            float bb_19 = s_19 - hi_19;
            float err_19 = hi_19 - (s_19 - bb_19) + (acc[3] - bb_19);
            pairs[2 * (rg * 6144 + 1152 + tir) + 1] = pairs[2 * (rg * 6144 + 1152 + tir) + 1] + err_19;
            pairs[2 * (rg * 6144 + 1152 + tir)] = s_19;
            acc[3] = 0.0f;
            float hi_20 = pairs[2 * (rg * 6144 + 1536 + tir)];
            float s_20 = hi_20 + acc[4];
            float bb_20 = s_20 - hi_20;
            float err_20 = hi_20 - (s_20 - bb_20) + (acc[4] - bb_20);
            pairs[2 * (rg * 6144 + 1536 + tir) + 1] = pairs[2 * (rg * 6144 + 1536 + tir) + 1] + err_20;
            pairs[2 * (rg * 6144 + 1536 + tir)] = s_20;
            acc[4] = 0.0f;
            float hi_21 = pairs[2 * (rg * 6144 + 1920 + tir)];
            float s_21 = hi_21 + acc[5];
            float bb_21 = s_21 - hi_21;
            float err_21 = hi_21 - (s_21 - bb_21) + (acc[5] - bb_21);
            pairs[2 * (rg * 6144 + 1920 + tir) + 1] = pairs[2 * (rg * 6144 + 1920 + tir) + 1] + err_21;
            pairs[2 * (rg * 6144 + 1920 + tir)] = s_21;
            acc[5] = 0.0f;
            float hi_22 = pairs[2 * (rg * 6144 + 2304 + tir)];
            float s_22 = hi_22 + acc[6];
            float bb_22 = s_22 - hi_22;
            float err_22 = hi_22 - (s_22 - bb_22) + (acc[6] - bb_22);
            pairs[2 * (rg * 6144 + 2304 + tir) + 1] = pairs[2 * (rg * 6144 + 2304 + tir) + 1] + err_22;
            pairs[2 * (rg * 6144 + 2304 + tir)] = s_22;
            acc[6] = 0.0f;
            float hi_23 = pairs[2 * (rg * 6144 + 2688 + tir)];
            float s_23 = hi_23 + acc[7];
            float bb_23 = s_23 - hi_23;
            float err_23 = hi_23 - (s_23 - bb_23) + (acc[7] - bb_23);
            pairs[2 * (rg * 6144 + 2688 + tir) + 1] = pairs[2 * (rg * 6144 + 2688 + tir) + 1] + err_23;
            pairs[2 * (rg * 6144 + 2688 + tir)] = s_23;
            acc[7] = 0.0f;
            float hi_24 = pairs[2 * (rg * 6144 + 3072 + tir)];
            float s_24 = hi_24 + acc[8];
            float bb_24 = s_24 - hi_24;
            float err_24 = hi_24 - (s_24 - bb_24) + (acc[8] - bb_24);
            pairs[2 * (rg * 6144 + 3072 + tir) + 1] = pairs[2 * (rg * 6144 + 3072 + tir) + 1] + err_24;
            pairs[2 * (rg * 6144 + 3072 + tir)] = s_24;
            acc[8] = 0.0f;
            float hi_25 = pairs[2 * (rg * 6144 + 3456 + tir)];
            float s_25 = hi_25 + acc[9];
            float bb_25 = s_25 - hi_25;
            float err_25 = hi_25 - (s_25 - bb_25) + (acc[9] - bb_25);
            pairs[2 * (rg * 6144 + 3456 + tir) + 1] = pairs[2 * (rg * 6144 + 3456 + tir) + 1] + err_25;
            pairs[2 * (rg * 6144 + 3456 + tir)] = s_25;
            acc[9] = 0.0f;
            float hi_26 = pairs[2 * (rg * 6144 + 3840 + tir)];
            float s_26 = hi_26 + acc[10];
            float bb_26 = s_26 - hi_26;
            float err_26 = hi_26 - (s_26 - bb_26) + (acc[10] - bb_26);
            pairs[2 * (rg * 6144 + 3840 + tir) + 1] = pairs[2 * (rg * 6144 + 3840 + tir) + 1] + err_26;
            pairs[2 * (rg * 6144 + 3840 + tir)] = s_26;
            acc[10] = 0.0f;
            float hi_27 = pairs[2 * (rg * 6144 + 4224 + tir)];
            float s_27 = hi_27 + acc[11];
            float bb_27 = s_27 - hi_27;
            float err_27 = hi_27 - (s_27 - bb_27) + (acc[11] - bb_27);
            pairs[2 * (rg * 6144 + 4224 + tir) + 1] = pairs[2 * (rg * 6144 + 4224 + tir) + 1] + err_27;
            pairs[2 * (rg * 6144 + 4224 + tir)] = s_27;
            acc[11] = 0.0f;
            float hi_28 = pairs[2 * (rg * 6144 + 4608 + tir)];
            float s_28 = hi_28 + acc[12];
            float bb_28 = s_28 - hi_28;
            float err_28 = hi_28 - (s_28 - bb_28) + (acc[12] - bb_28);
            pairs[2 * (rg * 6144 + 4608 + tir) + 1] = pairs[2 * (rg * 6144 + 4608 + tir) + 1] + err_28;
            pairs[2 * (rg * 6144 + 4608 + tir)] = s_28;
            acc[12] = 0.0f;
            float hi_29 = pairs[2 * (rg * 6144 + 4992 + tir)];
            float s_29 = hi_29 + acc[13];
            float bb_29 = s_29 - hi_29;
            float err_29 = hi_29 - (s_29 - bb_29) + (acc[13] - bb_29);
            pairs[2 * (rg * 6144 + 4992 + tir) + 1] = pairs[2 * (rg * 6144 + 4992 + tir) + 1] + err_29;
            pairs[2 * (rg * 6144 + 4992 + tir)] = s_29;
            acc[13] = 0.0f;
            float hi_30 = pairs[2 * (rg * 6144 + 5376 + tir)];
            float s_30 = hi_30 + acc[14];
            float bb_30 = s_30 - hi_30;
            float err_30 = hi_30 - (s_30 - bb_30) + (acc[14] - bb_30);
            pairs[2 * (rg * 6144 + 5376 + tir) + 1] = pairs[2 * (rg * 6144 + 5376 + tir) + 1] + err_30;
            pairs[2 * (rg * 6144 + 5376 + tir)] = s_30;
            acc[14] = 0.0f;
            float hi_31 = pairs[2 * (rg * 6144 + 5760 + tir)];
            float s_31 = hi_31 + acc[15];
            float bb_31 = s_31 - hi_31;
            float err_31 = hi_31 - (s_31 - bb_31) + (acc[15] - bb_31);
            pairs[2 * (rg * 6144 + 5760 + tir) + 1] = pairs[2 * (rg * 6144 + 5760 + tir) + 1] + err_31;
            pairs[2 * (rg * 6144 + 5760 + tir)] = s_31;
            acc[15] = 0.0f;
            double v64_0 = (double)pairs[2 * (rg * 6144 + tir)] + (double)pairs[2 * (rg * 6144 + tir) + 1];
            sums[rg * 6144 + tir] = v64_0;
            double v64_1 = (double)pairs[2 * (rg * 6144 + 384 + tir)] + (double)pairs[2 * (rg * 6144 + 384 + tir) + 1];
            sums[rg * 6144 + 384 + tir] = v64_1;
            double v64_2 = (double)pairs[2 * (rg * 6144 + 768 + tir)] + (double)pairs[2 * (rg * 6144 + 768 + tir) + 1];
            sums[rg * 6144 + 768 + tir] = v64_2;
            double v64_3 = (double)pairs[2 * (rg * 6144 + 1152 + tir)] + (double)pairs[2 * (rg * 6144 + 1152 + tir) + 1];
            sums[rg * 6144 + 1152 + tir] = v64_3;
            double v64_4 = (double)pairs[2 * (rg * 6144 + 1536 + tir)] + (double)pairs[2 * (rg * 6144 + 1536 + tir) + 1];
            sums[rg * 6144 + 1536 + tir] = v64_4;
            double v64_5 = (double)pairs[2 * (rg * 6144 + 1920 + tir)] + (double)pairs[2 * (rg * 6144 + 1920 + tir) + 1];
            sums[rg * 6144 + 1920 + tir] = v64_5;
            double v64_6 = (double)pairs[2 * (rg * 6144 + 2304 + tir)] + (double)pairs[2 * (rg * 6144 + 2304 + tir) + 1];
            sums[rg * 6144 + 2304 + tir] = v64_6;
            double v64_7 = (double)pairs[2 * (rg * 6144 + 2688 + tir)] + (double)pairs[2 * (rg * 6144 + 2688 + tir) + 1];
            sums[rg * 6144 + 2688 + tir] = v64_7;
            double v64_8 = (double)pairs[2 * (rg * 6144 + 3072 + tir)] + (double)pairs[2 * (rg * 6144 + 3072 + tir) + 1];
            sums[rg * 6144 + 3072 + tir] = v64_8;
            double v64_9 = (double)pairs[2 * (rg * 6144 + 3456 + tir)] + (double)pairs[2 * (rg * 6144 + 3456 + tir) + 1];
            sums[rg * 6144 + 3456 + tir] = v64_9;
            double v64_10 = (double)pairs[2 * (rg * 6144 + 3840 + tir)] + (double)pairs[2 * (rg * 6144 + 3840 + tir) + 1];
            sums[rg * 6144 + 3840 + tir] = v64_10;
            double v64_11 = (double)pairs[2 * (rg * 6144 + 4224 + tir)] + (double)pairs[2 * (rg * 6144 + 4224 + tir) + 1];
            sums[rg * 6144 + 4224 + tir] = v64_11;
            double v64_12 = (double)pairs[2 * (rg * 6144 + 4608 + tir)] + (double)pairs[2 * (rg * 6144 + 4608 + tir) + 1];
            sums[rg * 6144 + 4608 + tir] = v64_12;
            double v64_13 = (double)pairs[2 * (rg * 6144 + 4992 + tir)] + (double)pairs[2 * (rg * 6144 + 4992 + tir) + 1];
            sums[rg * 6144 + 4992 + tir] = v64_13;
            double v64_14 = (double)pairs[2 * (rg * 6144 + 5376 + tir)] + (double)pairs[2 * (rg * 6144 + 5376 + tir) + 1];
            sums[rg * 6144 + 5376 + tir] = v64_14;
            double v64_15 = (double)pairs[2 * (rg * 6144 + 5760 + tir)] + (double)pairs[2 * (rg * 6144 + 5760 + tir) + 1];
            sums[rg * 6144 + 5760 + tir] = v64_15;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (tid == 0) {
                if (first_c < n_chunks) {
                    {
                        void* _cpred_dst_9 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_9), "r"(sums_addr + 0), "r"((uint32_t)(12288))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_10 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 1536));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_10), "r"(sums_addr + 12288), "r"((uint32_t)(12288))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_11 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 3072));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_11), "r"(sums_addr + 24576), "r"((uint32_t)(12288))
                            : "memory");
                    }
                    {
                        void* _cpred_dst_12 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 4608));
                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                            " [%0], [%1], %2;"
                            :: "l"(_cpred_dst_12), "r"(sums_addr + 36864), "r"((uint32_t)(12288))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                unsigned int old;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(old) : "l"(&counters[1]), "r"(static_cast<uint32_t>(1)) : "memory");
                int last = 0;
                if (old + 1 == (unsigned int)num_bids) {
                    *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
                    last = 1;
                }
                flags[2] = last;
                if (c == n_chunks + (int)gridDim.x - 1) {
                    *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (flags[2] != 0) {
                double buf[32];
                {
                    uint64_t _scalar_bits_13;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_13) : "l"((const void*)(partial + (tid))) : "memory");
                    buf[0] = __longlong_as_double(_scalar_bits_13);
                }
                {
                    uint64_t _scalar_bits_14;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_14) : "l"((const void*)(partial + (6144 + tid))) : "memory");
                    buf[1] = __longlong_as_double(_scalar_bits_14);
                }
                {
                    uint64_t _scalar_bits_15;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_15) : "l"((const void*)(partial + (12288 + tid))) : "memory");
                    buf[2] = __longlong_as_double(_scalar_bits_15);
                }
                {
                    uint64_t _scalar_bits_16;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_16) : "l"((const void*)(partial + (18432 + tid))) : "memory");
                    buf[3] = __longlong_as_double(_scalar_bits_16);
                }
                {
                    uint64_t _scalar_bits_17;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_17) : "l"((const void*)(partial + (384 + tid))) : "memory");
                    buf[4] = __longlong_as_double(_scalar_bits_17);
                }
                {
                    uint64_t _scalar_bits_18;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_18) : "l"((const void*)(partial + (6528 + tid))) : "memory");
                    buf[5] = __longlong_as_double(_scalar_bits_18);
                }
                {
                    uint64_t _scalar_bits_19;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_19) : "l"((const void*)(partial + (12672 + tid))) : "memory");
                    buf[6] = __longlong_as_double(_scalar_bits_19);
                }
                {
                    uint64_t _scalar_bits_20;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_20) : "l"((const void*)(partial + (18816 + tid))) : "memory");
                    buf[7] = __longlong_as_double(_scalar_bits_20);
                }
                {
                    uint64_t _scalar_bits_21;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_21) : "l"((const void*)(partial + (768 + tid))) : "memory");
                    buf[8] = __longlong_as_double(_scalar_bits_21);
                }
                {
                    uint64_t _scalar_bits_22;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_22) : "l"((const void*)(partial + (6912 + tid))) : "memory");
                    buf[9] = __longlong_as_double(_scalar_bits_22);
                }
                {
                    uint64_t _scalar_bits_23;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_23) : "l"((const void*)(partial + (13056 + tid))) : "memory");
                    buf[10] = __longlong_as_double(_scalar_bits_23);
                }
                {
                    uint64_t _scalar_bits_24;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_24) : "l"((const void*)(partial + (19200 + tid))) : "memory");
                    buf[11] = __longlong_as_double(_scalar_bits_24);
                }
                {
                    uint64_t _scalar_bits_25;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_25) : "l"((const void*)(partial + (1152 + tid))) : "memory");
                    buf[12] = __longlong_as_double(_scalar_bits_25);
                }
                {
                    uint64_t _scalar_bits_26;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_26) : "l"((const void*)(partial + (7296 + tid))) : "memory");
                    buf[13] = __longlong_as_double(_scalar_bits_26);
                }
                {
                    uint64_t _scalar_bits_27;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_27) : "l"((const void*)(partial + (13440 + tid))) : "memory");
                    buf[14] = __longlong_as_double(_scalar_bits_27);
                }
                {
                    uint64_t _scalar_bits_28;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_28) : "l"((const void*)(partial + (19584 + tid))) : "memory");
                    buf[15] = __longlong_as_double(_scalar_bits_28);
                }
                {
                    uint64_t _scalar_bits_29;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_29) : "l"((const void*)(partial + (1536 + tid))) : "memory");
                    buf[16] = __longlong_as_double(_scalar_bits_29);
                }
                {
                    uint64_t _scalar_bits_30;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_30) : "l"((const void*)(partial + (7680 + tid))) : "memory");
                    buf[17] = __longlong_as_double(_scalar_bits_30);
                }
                {
                    uint64_t _scalar_bits_31;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_31) : "l"((const void*)(partial + (13824 + tid))) : "memory");
                    buf[18] = __longlong_as_double(_scalar_bits_31);
                }
                {
                    uint64_t _scalar_bits_32;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_32) : "l"((const void*)(partial + (19968 + tid))) : "memory");
                    buf[19] = __longlong_as_double(_scalar_bits_32);
                }
                {
                    uint64_t _scalar_bits_33;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_33) : "l"((const void*)(partial + (1920 + tid))) : "memory");
                    buf[20] = __longlong_as_double(_scalar_bits_33);
                }
                {
                    uint64_t _scalar_bits_34;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_34) : "l"((const void*)(partial + (8064 + tid))) : "memory");
                    buf[21] = __longlong_as_double(_scalar_bits_34);
                }
                {
                    uint64_t _scalar_bits_35;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_35) : "l"((const void*)(partial + (14208 + tid))) : "memory");
                    buf[22] = __longlong_as_double(_scalar_bits_35);
                }
                {
                    uint64_t _scalar_bits_36;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_36) : "l"((const void*)(partial + (20352 + tid))) : "memory");
                    buf[23] = __longlong_as_double(_scalar_bits_36);
                }
                {
                    uint64_t _scalar_bits_37;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_37) : "l"((const void*)(partial + (2304 + tid))) : "memory");
                    buf[24] = __longlong_as_double(_scalar_bits_37);
                }
                {
                    uint64_t _scalar_bits_38;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_38) : "l"((const void*)(partial + (8448 + tid))) : "memory");
                    buf[25] = __longlong_as_double(_scalar_bits_38);
                }
                {
                    uint64_t _scalar_bits_39;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_39) : "l"((const void*)(partial + (14592 + tid))) : "memory");
                    buf[26] = __longlong_as_double(_scalar_bits_39);
                }
                {
                    uint64_t _scalar_bits_40;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_40) : "l"((const void*)(partial + (20736 + tid))) : "memory");
                    buf[27] = __longlong_as_double(_scalar_bits_40);
                }
                {
                    uint64_t _scalar_bits_41;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_41) : "l"((const void*)(partial + (2688 + tid))) : "memory");
                    buf[28] = __longlong_as_double(_scalar_bits_41);
                }
                {
                    uint64_t _scalar_bits_42;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_42) : "l"((const void*)(partial + (8832 + tid))) : "memory");
                    buf[29] = __longlong_as_double(_scalar_bits_42);
                }
                {
                    uint64_t _scalar_bits_43;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_43) : "l"((const void*)(partial + (14976 + tid))) : "memory");
                    buf[30] = __longlong_as_double(_scalar_bits_43);
                }
                {
                    uint64_t _scalar_bits_44;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_44) : "l"((const void*)(partial + (21120 + tid))) : "memory");
                    buf[31] = __longlong_as_double(_scalar_bits_44);
                }
                double tot_0 = buf[0];
                tot_0 = tot_0 + buf[1];
                tot_0 = tot_0 + buf[2];
                tot_0 = tot_0 + buf[3];
                int at_w_0 = tid;
                dwbuf[(at_w_0 / 384 / 8 * 384 + at_w_0 % 384) * 8 + at_w_0 / 384 % 8] = (float)tot_0;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_0)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_1 = buf[4];
                tot_1 = tot_1 + buf[5];
                tot_1 = tot_1 + buf[6];
                tot_1 = tot_1 + buf[7];
                int at_w_1 = 384 + tid;
                dwbuf[(at_w_1 / 384 / 8 * 384 + at_w_1 % 384) * 8 + at_w_1 / 384 % 8] = (float)tot_1;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_1)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_2 = buf[8];
                tot_2 = tot_2 + buf[9];
                tot_2 = tot_2 + buf[10];
                tot_2 = tot_2 + buf[11];
                int at_w_2 = 768 + tid;
                dwbuf[(at_w_2 / 384 / 8 * 384 + at_w_2 % 384) * 8 + at_w_2 / 384 % 8] = (float)tot_2;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_2)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_2)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_2)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_2)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_3 = buf[12];
                tot_3 = tot_3 + buf[13];
                tot_3 = tot_3 + buf[14];
                tot_3 = tot_3 + buf[15];
                int at_w_3 = 1152 + tid;
                dwbuf[(at_w_3 / 384 / 8 * 384 + at_w_3 % 384) * 8 + at_w_3 / 384 % 8] = (float)tot_3;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_3)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_3)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_3)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_3)) = (unsigned long long)((unsigned long long)(0.0));
                {
                    uint64_t _scalar_bits_45;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_45) : "l"((const void*)(partial + (3072 + tid))) : "memory");
                    buf[0] = __longlong_as_double(_scalar_bits_45);
                }
                {
                    uint64_t _scalar_bits_46;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_46) : "l"((const void*)(partial + (9216 + tid))) : "memory");
                    buf[1] = __longlong_as_double(_scalar_bits_46);
                }
                {
                    uint64_t _scalar_bits_47;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_47) : "l"((const void*)(partial + (15360 + tid))) : "memory");
                    buf[2] = __longlong_as_double(_scalar_bits_47);
                }
                {
                    uint64_t _scalar_bits_48;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_48) : "l"((const void*)(partial + (21504 + tid))) : "memory");
                    buf[3] = __longlong_as_double(_scalar_bits_48);
                }
                {
                    uint64_t _scalar_bits_49;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_49) : "l"((const void*)(partial + (3456 + tid))) : "memory");
                    buf[4] = __longlong_as_double(_scalar_bits_49);
                }
                {
                    uint64_t _scalar_bits_50;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_50) : "l"((const void*)(partial + (9600 + tid))) : "memory");
                    buf[5] = __longlong_as_double(_scalar_bits_50);
                }
                {
                    uint64_t _scalar_bits_51;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_51) : "l"((const void*)(partial + (15744 + tid))) : "memory");
                    buf[6] = __longlong_as_double(_scalar_bits_51);
                }
                {
                    uint64_t _scalar_bits_52;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_52) : "l"((const void*)(partial + (21888 + tid))) : "memory");
                    buf[7] = __longlong_as_double(_scalar_bits_52);
                }
                {
                    uint64_t _scalar_bits_53;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_53) : "l"((const void*)(partial + (3840 + tid))) : "memory");
                    buf[8] = __longlong_as_double(_scalar_bits_53);
                }
                {
                    uint64_t _scalar_bits_54;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_54) : "l"((const void*)(partial + (9984 + tid))) : "memory");
                    buf[9] = __longlong_as_double(_scalar_bits_54);
                }
                {
                    uint64_t _scalar_bits_55;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_55) : "l"((const void*)(partial + (16128 + tid))) : "memory");
                    buf[10] = __longlong_as_double(_scalar_bits_55);
                }
                {
                    uint64_t _scalar_bits_56;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_56) : "l"((const void*)(partial + (22272 + tid))) : "memory");
                    buf[11] = __longlong_as_double(_scalar_bits_56);
                }
                {
                    uint64_t _scalar_bits_57;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_57) : "l"((const void*)(partial + (4224 + tid))) : "memory");
                    buf[12] = __longlong_as_double(_scalar_bits_57);
                }
                {
                    uint64_t _scalar_bits_58;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_58) : "l"((const void*)(partial + (10368 + tid))) : "memory");
                    buf[13] = __longlong_as_double(_scalar_bits_58);
                }
                {
                    uint64_t _scalar_bits_59;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_59) : "l"((const void*)(partial + (16512 + tid))) : "memory");
                    buf[14] = __longlong_as_double(_scalar_bits_59);
                }
                {
                    uint64_t _scalar_bits_60;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_60) : "l"((const void*)(partial + (22656 + tid))) : "memory");
                    buf[15] = __longlong_as_double(_scalar_bits_60);
                }
                double tot_4 = buf[16];
                tot_4 = tot_4 + buf[17];
                tot_4 = tot_4 + buf[18];
                tot_4 = tot_4 + buf[19];
                int at_w_4 = 1536 + tid;
                dwbuf[(at_w_4 / 384 / 8 * 384 + at_w_4 % 384) * 8 + at_w_4 / 384 % 8] = (float)tot_4;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_4)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_4)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_4)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_4)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_5 = buf[20];
                tot_5 = tot_5 + buf[21];
                tot_5 = tot_5 + buf[22];
                tot_5 = tot_5 + buf[23];
                int at_w_5 = 1920 + tid;
                dwbuf[(at_w_5 / 384 / 8 * 384 + at_w_5 % 384) * 8 + at_w_5 / 384 % 8] = (float)tot_5;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_5)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_5)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_5)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_5)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_6 = buf[24];
                tot_6 = tot_6 + buf[25];
                tot_6 = tot_6 + buf[26];
                tot_6 = tot_6 + buf[27];
                int at_w_6 = 2304 + tid;
                dwbuf[(at_w_6 / 384 / 8 * 384 + at_w_6 % 384) * 8 + at_w_6 / 384 % 8] = (float)tot_6;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_6)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_6)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_6)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_6)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_7 = buf[28];
                tot_7 = tot_7 + buf[29];
                tot_7 = tot_7 + buf[30];
                tot_7 = tot_7 + buf[31];
                int at_w_7 = 2688 + tid;
                dwbuf[(at_w_7 / 384 / 8 * 384 + at_w_7 % 384) * 8 + at_w_7 / 384 % 8] = (float)tot_7;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_7)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_7)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_7)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_7)) = (unsigned long long)((unsigned long long)(0.0));
                {
                    uint64_t _scalar_bits_61;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_61) : "l"((const void*)(partial + (4608 + tid))) : "memory");
                    buf[16] = __longlong_as_double(_scalar_bits_61);
                }
                {
                    uint64_t _scalar_bits_62;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_62) : "l"((const void*)(partial + (10752 + tid))) : "memory");
                    buf[17] = __longlong_as_double(_scalar_bits_62);
                }
                {
                    uint64_t _scalar_bits_63;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_63) : "l"((const void*)(partial + (16896 + tid))) : "memory");
                    buf[18] = __longlong_as_double(_scalar_bits_63);
                }
                {
                    uint64_t _scalar_bits_64;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_64) : "l"((const void*)(partial + (23040 + tid))) : "memory");
                    buf[19] = __longlong_as_double(_scalar_bits_64);
                }
                {
                    uint64_t _scalar_bits_65;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_65) : "l"((const void*)(partial + (4992 + tid))) : "memory");
                    buf[20] = __longlong_as_double(_scalar_bits_65);
                }
                {
                    uint64_t _scalar_bits_66;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_66) : "l"((const void*)(partial + (11136 + tid))) : "memory");
                    buf[21] = __longlong_as_double(_scalar_bits_66);
                }
                {
                    uint64_t _scalar_bits_67;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_67) : "l"((const void*)(partial + (17280 + tid))) : "memory");
                    buf[22] = __longlong_as_double(_scalar_bits_67);
                }
                {
                    uint64_t _scalar_bits_68;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_68) : "l"((const void*)(partial + (23424 + tid))) : "memory");
                    buf[23] = __longlong_as_double(_scalar_bits_68);
                }
                {
                    uint64_t _scalar_bits_69;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_69) : "l"((const void*)(partial + (5376 + tid))) : "memory");
                    buf[24] = __longlong_as_double(_scalar_bits_69);
                }
                {
                    uint64_t _scalar_bits_70;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_70) : "l"((const void*)(partial + (11520 + tid))) : "memory");
                    buf[25] = __longlong_as_double(_scalar_bits_70);
                }
                {
                    uint64_t _scalar_bits_71;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_71) : "l"((const void*)(partial + (17664 + tid))) : "memory");
                    buf[26] = __longlong_as_double(_scalar_bits_71);
                }
                {
                    uint64_t _scalar_bits_72;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_72) : "l"((const void*)(partial + (23808 + tid))) : "memory");
                    buf[27] = __longlong_as_double(_scalar_bits_72);
                }
                {
                    uint64_t _scalar_bits_73;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_73) : "l"((const void*)(partial + (5760 + tid))) : "memory");
                    buf[28] = __longlong_as_double(_scalar_bits_73);
                }
                {
                    uint64_t _scalar_bits_74;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_74) : "l"((const void*)(partial + (11904 + tid))) : "memory");
                    buf[29] = __longlong_as_double(_scalar_bits_74);
                }
                {
                    uint64_t _scalar_bits_75;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_75) : "l"((const void*)(partial + (18048 + tid))) : "memory");
                    buf[30] = __longlong_as_double(_scalar_bits_75);
                }
                {
                    uint64_t _scalar_bits_76;
                    asm volatile("ld.global.cg.b64 %0, [%1];"
                        : "=l"(_scalar_bits_76) : "l"((const void*)(partial + (24192 + tid))) : "memory");
                    buf[31] = __longlong_as_double(_scalar_bits_76);
                }
                double tot_8 = buf[0];
                tot_8 = tot_8 + buf[1];
                tot_8 = tot_8 + buf[2];
                tot_8 = tot_8 + buf[3];
                int at_w_8 = 3072 + tid;
                dwbuf[(at_w_8 / 384 / 8 * 384 + at_w_8 % 384) * 8 + at_w_8 / 384 % 8] = (float)tot_8;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_8)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_8)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_8)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_8)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_9 = buf[4];
                tot_9 = tot_9 + buf[5];
                tot_9 = tot_9 + buf[6];
                tot_9 = tot_9 + buf[7];
                int at_w_9 = 3456 + tid;
                dwbuf[(at_w_9 / 384 / 8 * 384 + at_w_9 % 384) * 8 + at_w_9 / 384 % 8] = (float)tot_9;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_9)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_9)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_9)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_9)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_10 = buf[8];
                tot_10 = tot_10 + buf[9];
                tot_10 = tot_10 + buf[10];
                tot_10 = tot_10 + buf[11];
                int at_w_10 = 3840 + tid;
                dwbuf[(at_w_10 / 384 / 8 * 384 + at_w_10 % 384) * 8 + at_w_10 / 384 % 8] = (float)tot_10;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_10)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_10)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_10)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_10)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_11 = buf[12];
                tot_11 = tot_11 + buf[13];
                tot_11 = tot_11 + buf[14];
                tot_11 = tot_11 + buf[15];
                int at_w_11 = 4224 + tid;
                dwbuf[(at_w_11 / 384 / 8 * 384 + at_w_11 % 384) * 8 + at_w_11 / 384 % 8] = (float)tot_11;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_11)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_11)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_11)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_11)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_12 = buf[16];
                tot_12 = tot_12 + buf[17];
                tot_12 = tot_12 + buf[18];
                tot_12 = tot_12 + buf[19];
                int at_w_12 = 4608 + tid;
                dwbuf[(at_w_12 / 384 / 8 * 384 + at_w_12 % 384) * 8 + at_w_12 / 384 % 8] = (float)tot_12;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_12)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_12)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_12)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_12)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_13 = buf[20];
                tot_13 = tot_13 + buf[21];
                tot_13 = tot_13 + buf[22];
                tot_13 = tot_13 + buf[23];
                int at_w_13 = 4992 + tid;
                dwbuf[(at_w_13 / 384 / 8 * 384 + at_w_13 % 384) * 8 + at_w_13 / 384 % 8] = (float)tot_13;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_13)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_13)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_13)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_13)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_14 = buf[24];
                tot_14 = tot_14 + buf[25];
                tot_14 = tot_14 + buf[26];
                tot_14 = tot_14 + buf[27];
                int at_w_14 = 5376 + tid;
                dwbuf[(at_w_14 / 384 / 8 * 384 + at_w_14 % 384) * 8 + at_w_14 / 384 % 8] = (float)tot_14;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_14)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_14)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_14)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_14)) = (unsigned long long)((unsigned long long)(0.0));
                double tot_15 = buf[28];
                tot_15 = tot_15 + buf[29];
                tot_15 = tot_15 + buf[30];
                tot_15 = tot_15 + buf[31];
                int at_w_15 = 5760 + tid;
                dwbuf[(at_w_15 / 384 / 8 * 384 + at_w_15 % 384) * 8 + at_w_15 / 384 % 8] = (float)tot_15;
                *reinterpret_cast<unsigned long long*>(partial + (at_w_15)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (6144 + at_w_15)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (12288 + at_w_15)) = (unsigned long long)((unsigned long long)(0.0));
                *reinterpret_cast<unsigned long long*>(partial + (18432 + at_w_15)) = (unsigned long long)((unsigned long long)(0.0));
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                int col_0 = tid * 4;
                if (col_0 < 6144) {
                    float o4_0[4];
                    o4_0[0] = dwbuf[col_0];
                    o4_0[1] = dwbuf[col_0 + 1];
                    o4_0[2] = dwbuf[col_0 + 2];
                    o4_0[3] = dwbuf[col_0 + 3];
                    {
                        float4 _v4 = make_float4(o4_0[0 + 0], o4_0[0 + 1], o4_0[0 + 2], o4_0[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_0) = _v4;
                    }
                }
                int col_1 = (384 + tid) * 4;
                if (col_1 < 6144) {
                    float o4_1[4];
                    o4_1[0] = dwbuf[col_1];
                    o4_1[1] = dwbuf[col_1 + 1];
                    o4_1[2] = dwbuf[col_1 + 2];
                    o4_1[3] = dwbuf[col_1 + 3];
                    {
                        float4 _v4 = make_float4(o4_1[0 + 0], o4_1[0 + 1], o4_1[0 + 2], o4_1[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_1) = _v4;
                    }
                }
                int col_2 = (768 + tid) * 4;
                if (col_2 < 6144) {
                    float o4_2[4];
                    o4_2[0] = dwbuf[col_2];
                    o4_2[1] = dwbuf[col_2 + 1];
                    o4_2[2] = dwbuf[col_2 + 2];
                    o4_2[3] = dwbuf[col_2 + 3];
                    {
                        float4 _v4 = make_float4(o4_2[0 + 0], o4_2[0 + 1], o4_2[0 + 2], o4_2[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_2) = _v4;
                    }
                }
                int col_3 = (1152 + tid) * 4;
                if (col_3 < 6144) {
                    float o4_3[4];
                    o4_3[0] = dwbuf[col_3];
                    o4_3[1] = dwbuf[col_3 + 1];
                    o4_3[2] = dwbuf[col_3 + 2];
                    o4_3[3] = dwbuf[col_3 + 3];
                    {
                        float4 _v4 = make_float4(o4_3[0 + 0], o4_3[0 + 1], o4_3[0 + 2], o4_3[0 + 3]);
                        *reinterpret_cast<float4*>(dw + col_3) = _v4;
                    }
                }
            }
        }
    }
    // ---- Role: prod ----
    if (warp == 12) {
        { // prod_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            if (elect_sync()) {
                unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
                int p_c = (int)_atomic_fetch_add_0;
                int p_first = p_c;
                int p_row = 0;
                int p_end = 0;
                unsigned int p_nxt = 0;
                if (p_c < n_chunks) {
                    int stages = T;
                    int sl = rows_per_chunk;
                    if (sl > stages) {
                        sl = stages;
                    }
                    if (sl < 1) {
                        sl = 1;
                    }
                    int share = (stages + sl - 1) / sl;
                    int tail = (share + 11) / 12;
                    if (tail < 1) {
                        tail = 1;
                    }
                    int rem = stages;
                    int size = (share + 1) / 2;
                    int first = 0;
                    int base = 0;
                    int span = 0;
                    int halving = 1;
                    int begin_it = 0;
                    #pragma unroll 1
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving != 0) {
                            if (size > tail) {
                                if (rem >= sl * size) {
                                    if (p_c < first + sl) {
                                        span = size;
                                        begin_it = base + (p_c - first) * size;
                                        halving = 0;
                                    } else {
                                        first = first + sl;
                                        base = base + sl * size;
                                        rem = rem - sl * size;
                                        size = (size + 1) / 2;
                                    }
                                } else {
                                    halving = 0;
                                }
                            } else {
                                halving = 0;
                            }
                        }
                    }
                    if (span == 0) {
                        int quot = rem / sl;
                        int extra = rem - quot * sl;
                        if (p_c < first + extra) {
                            span = quot + 1;
                            begin_it = base + (p_c - first) * (quot + 1);
                        } else if (p_c < first + sl) {
                            span = quot;
                            begin_it = base + extra * (quot + 1) + (p_c - first - extra) * quot;
                        } else {
                            begin_it = stages;
                        }
                    }
                    int _min_1 = ((begin_it) < (T) ? (begin_it) : (T));
                    p_row = _min_1;
                    int _min_2 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
                    p_end = _min_2;
                    unsigned int _atomic_fetch_add_3 = atomicAdd(&counters[0], 1);
                    p_nxt = _atomic_fetch_add_3;
                }
                int p_done = 0;
                int p_ring = (int)ring_addr;
                #pragma unroll 1
                for (int pk = 0; pk < T + n_chunks + 1; pk++) {
                    int _stage_empty = (int)((unsigned int)pk % 4);
                    int _parity_empty = (int)((unsigned int)pk / 4 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int slot = pk % 4;
                    if (p_row >= p_end) {
                        if (p_c < n_chunks) {
                            p_c = (int)p_nxt;
                            if (p_c < n_chunks) {
                                int stages_1 = T;
                                int sl_1 = rows_per_chunk;
                                if (sl_1 > stages_1) {
                                    sl_1 = stages_1;
                                }
                                if (sl_1 < 1) {
                                    sl_1 = 1;
                                }
                                int share_1 = (stages_1 + sl_1 - 1) / sl_1;
                                int tail_1 = (share_1 + 11) / 12;
                                if (tail_1 < 1) {
                                    tail_1 = 1;
                                }
                                int rem_1 = stages_1;
                                int size_1 = (share_1 + 1) / 2;
                                int first_1 = 0;
                                int base_1 = 0;
                                int span_1 = 0;
                                int halving_1 = 1;
                                int begin_it_1 = 0;
                                #pragma unroll 1
                                for (int k_2 = 0; k_2 < 5; k_2++) {
                                    if (halving_1 != 0) {
                                        if (size_1 > tail_1) {
                                            if (rem_1 >= sl_1 * size_1) {
                                                if (p_c < first_1 + sl_1) {
                                                    span_1 = size_1;
                                                    begin_it_1 = base_1 + (p_c - first_1) * size_1;
                                                    halving_1 = 0;
                                                } else {
                                                    first_1 = first_1 + sl_1;
                                                    base_1 = base_1 + sl_1 * size_1;
                                                    rem_1 = rem_1 - sl_1 * size_1;
                                                    size_1 = (size_1 + 1) / 2;
                                                }
                                            } else {
                                                halving_1 = 0;
                                            }
                                        } else {
                                            halving_1 = 0;
                                        }
                                    }
                                }
                                if (span_1 == 0) {
                                    int quot_1 = rem_1 / sl_1;
                                    int extra_1 = rem_1 - quot_1 * sl_1;
                                    if (p_c < first_1 + extra_1) {
                                        span_1 = quot_1 + 1;
                                        begin_it_1 = base_1 + (p_c - first_1) * (quot_1 + 1);
                                    } else if (p_c < first_1 + sl_1) {
                                        span_1 = quot_1;
                                        begin_it_1 = base_1 + extra_1 * (quot_1 + 1) + (p_c - first_1 - extra_1) * quot_1;
                                    } else {
                                        begin_it_1 = stages_1;
                                    }
                                }
                                int _min_4 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                                p_row = _min_4;
                                int _min_5 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                                p_end = _min_5;
                                unsigned int _atomic_fetch_add_6 = atomicAdd(&counters[0], 1);
                                p_nxt = _atomic_fetch_add_6;
                            }
                        }
                    }
                    int nval_1 = 0;
                    if (p_row < p_end) {
                        int _min_7 = ((p_end - p_row) < (1) ? (p_end - p_row) : (1));
                        nval_1 = _min_7;
                    }
                    if (nval_1 > 0) {
                        meta[2 * slot] = p_row;
                        meta[2 * slot + 1] = nval_1;
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 36864);
                        int sbase = p_ring + slot * 36864;
                        int _min_8 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
                        long long row_i_0 = (long long)(p_row + _min_8);
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + row_i_0 * g_h_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 4)) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                    } else {
                        meta[2 * slot] = p_c;
                        meta[2 * slot + 1] = -1 - p_first;
                        mbarrier_arrive(full_addr + ((int)((unsigned int)pk % 4)) * 8);
                        p_done = 1;
                    }
                    p_row = p_row + nval_1;
                    if (p_done != 0) {
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
