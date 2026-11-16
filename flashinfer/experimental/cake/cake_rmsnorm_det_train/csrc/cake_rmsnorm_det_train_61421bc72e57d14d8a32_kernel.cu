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

__global__ void
kernel_cake_rmsnorm_det_train_61421bc72e57d14d8a32(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    #define empty_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* dots = reinterpret_cast<float*>(smem_raw + 1024);
    const int dots_addr = smem + 1024;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1152);
    const int ring_addr = smem + 1152;

    // Mbarrier init (6 barriers at smem_raw[0..48))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // empty: 3 barriers, init_count=192
            mbarrier_init(smem + 24, 192);
            mbarrier_init(smem + 32, 192);
            mbarrier_init(smem + 40, 192);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: compute ----
    if (warp <= 5) {
        { // compute_main
            int t = tid;
            int ring_at = (int)ring_addr;
            unsigned int ww[16];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)((0 + t) * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)((192 + t) * 8));
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[4]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((384 + t) * 8));
                uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[8]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_2[_blk] = _vptr_2[_blk];
                }
            }
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(w + (unsigned long long)((576 + t) * 8));
                uint4* _vdst_3 = reinterpret_cast<uint4*>(&ww[12]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_3[_blk] = _vptr_3[_blk];
                }
            }
            float acc[32];
            unsigned int gw[16];
            unsigned int xw[16];
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
                acc[16] = 0.0f;
                acc[17] = 0.0f;
                acc[18] = 0.0f;
                acc[19] = 0.0f;
                acc[20] = 0.0f;
                acc[21] = 0.0f;
                acc[22] = 0.0f;
                acc[23] = 0.0f;
                acc[24] = 0.0f;
                acc[25] = 0.0f;
                acc[26] = 0.0f;
                acc[27] = 0.0f;
                acc[28] = 0.0f;
                acc[29] = 0.0f;
                acc[30] = 0.0f;
                acc[31] = 0.0f;
                int row = slot;
                while (row < T) {
                    float rq = r[row];
                    int _stage_full = (int)((unsigned int)k % 3);
                    int _parity_full = (int)((unsigned int)k / 3 % 2);
                    mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                    int sbase = ring_at + k % 3 * 24576;
                    __int128_t _smem_raw_4;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(sbase + (0 + t) * 8 * 2));
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                    __int128_t _smem_raw_5;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_5) : "r"(sbase + 12288 + (0 + t) * 8 * 2));
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_5)[3];
                    __int128_t _smem_raw_6;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_6) : "r"(sbase + (192 + t) * 8 * 2));
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_6)[3];
                    __int128_t _smem_raw_7;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_7) : "r"(sbase + 12288 + (192 + t) * 8 * 2));
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_7)[3];
                    __int128_t _smem_raw_8;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_8) : "r"(sbase + (384 + t) * 8 * 2));
                    gw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[0];
                    gw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[1];
                    gw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[2];
                    gw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_8)[3];
                    __int128_t _smem_raw_9;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_9) : "r"(sbase + 12288 + (384 + t) * 8 * 2));
                    xw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[0];
                    xw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[1];
                    xw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[2];
                    xw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_9)[3];
                    __int128_t _smem_raw_10;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_10) : "r"(sbase + (576 + t) * 8 * 2));
                    gw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[0];
                    gw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[1];
                    gw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[2];
                    gw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_10)[3];
                    __int128_t _smem_raw_11;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sbase + 12288 + (576 + t) * 8 * 2));
                    xw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                    xw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                    xw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                    xw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                    float dq = 0.0f;
                    float dqv[2];
                    dqv[0] = 0.0f;
                    dqv[1] = 0.0f;
                    float ga_0 = __uint_as_float(gw[0] << 16);
                    float gb_0 = __uint_as_float(gw[0] & 4294901760u);
                    float xa_0 = __uint_as_float(xw[0] << 16);
                    float xb_0 = __uint_as_float(xw[0] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_0 * __uint_as_float(ww[0] << 16), xa_0, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_0 * __uint_as_float(ww[0] & 4294901760u), xb_0, dqv[0]);
                    acc[0] = __fmaf_rn(ga_0 * xa_0, rq, acc[0]);
                    acc[1] = __fmaf_rn(gb_0 * xb_0, rq, acc[1]);
                    float ga_1 = __uint_as_float(gw[1] << 16);
                    float gb_1 = __uint_as_float(gw[1] & 4294901760u);
                    float xa_1 = __uint_as_float(xw[1] << 16);
                    float xb_1 = __uint_as_float(xw[1] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_1 * __uint_as_float(ww[1] << 16), xa_1, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_1 * __uint_as_float(ww[1] & 4294901760u), xb_1, dqv[0]);
                    acc[2] = __fmaf_rn(ga_1 * xa_1, rq, acc[2]);
                    acc[3] = __fmaf_rn(gb_1 * xb_1, rq, acc[3]);
                    float ga_2 = __uint_as_float(gw[2] << 16);
                    float gb_2 = __uint_as_float(gw[2] & 4294901760u);
                    float xa_2 = __uint_as_float(xw[2] << 16);
                    float xb_2 = __uint_as_float(xw[2] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_2 * __uint_as_float(ww[2] << 16), xa_2, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_2 * __uint_as_float(ww[2] & 4294901760u), xb_2, dqv[0]);
                    acc[4] = __fmaf_rn(ga_2 * xa_2, rq, acc[4]);
                    acc[5] = __fmaf_rn(gb_2 * xb_2, rq, acc[5]);
                    float ga_3 = __uint_as_float(gw[3] << 16);
                    float gb_3 = __uint_as_float(gw[3] & 4294901760u);
                    float xa_3 = __uint_as_float(xw[3] << 16);
                    float xb_3 = __uint_as_float(xw[3] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_3 * __uint_as_float(ww[3] << 16), xa_3, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_3 * __uint_as_float(ww[3] & 4294901760u), xb_3, dqv[0]);
                    acc[6] = __fmaf_rn(ga_3 * xa_3, rq, acc[6]);
                    acc[7] = __fmaf_rn(gb_3 * xb_3, rq, acc[7]);
                    float ga_4 = __uint_as_float(gw[4] << 16);
                    float gb_4 = __uint_as_float(gw[4] & 4294901760u);
                    float xa_4 = __uint_as_float(xw[4] << 16);
                    float xb_4 = __uint_as_float(xw[4] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_4 * __uint_as_float(ww[4] << 16), xa_4, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_4 * __uint_as_float(ww[4] & 4294901760u), xb_4, dqv[1]);
                    acc[8] = __fmaf_rn(ga_4 * xa_4, rq, acc[8]);
                    acc[9] = __fmaf_rn(gb_4 * xb_4, rq, acc[9]);
                    float ga_5 = __uint_as_float(gw[5] << 16);
                    float gb_5 = __uint_as_float(gw[5] & 4294901760u);
                    float xa_5 = __uint_as_float(xw[5] << 16);
                    float xb_5 = __uint_as_float(xw[5] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_5 * __uint_as_float(ww[5] << 16), xa_5, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_5 * __uint_as_float(ww[5] & 4294901760u), xb_5, dqv[1]);
                    acc[10] = __fmaf_rn(ga_5 * xa_5, rq, acc[10]);
                    acc[11] = __fmaf_rn(gb_5 * xb_5, rq, acc[11]);
                    float ga_6 = __uint_as_float(gw[6] << 16);
                    float gb_6 = __uint_as_float(gw[6] & 4294901760u);
                    float xa_6 = __uint_as_float(xw[6] << 16);
                    float xb_6 = __uint_as_float(xw[6] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_6 * __uint_as_float(ww[6] << 16), xa_6, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_6 * __uint_as_float(ww[6] & 4294901760u), xb_6, dqv[1]);
                    acc[12] = __fmaf_rn(ga_6 * xa_6, rq, acc[12]);
                    acc[13] = __fmaf_rn(gb_6 * xb_6, rq, acc[13]);
                    float ga_7 = __uint_as_float(gw[7] << 16);
                    float gb_7 = __uint_as_float(gw[7] & 4294901760u);
                    float xa_7 = __uint_as_float(xw[7] << 16);
                    float xb_7 = __uint_as_float(xw[7] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_7 * __uint_as_float(ww[7] << 16), xa_7, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_7 * __uint_as_float(ww[7] & 4294901760u), xb_7, dqv[1]);
                    acc[14] = __fmaf_rn(ga_7 * xa_7, rq, acc[14]);
                    acc[15] = __fmaf_rn(gb_7 * xb_7, rq, acc[15]);
                    float ga_8 = __uint_as_float(gw[8] << 16);
                    float gb_8 = __uint_as_float(gw[8] & 4294901760u);
                    float xa_8 = __uint_as_float(xw[8] << 16);
                    float xb_8 = __uint_as_float(xw[8] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_8 * __uint_as_float(ww[8] << 16), xa_8, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_8 * __uint_as_float(ww[8] & 4294901760u), xb_8, dqv[0]);
                    acc[16] = __fmaf_rn(ga_8 * xa_8, rq, acc[16]);
                    acc[17] = __fmaf_rn(gb_8 * xb_8, rq, acc[17]);
                    float ga_9 = __uint_as_float(gw[9] << 16);
                    float gb_9 = __uint_as_float(gw[9] & 4294901760u);
                    float xa_9 = __uint_as_float(xw[9] << 16);
                    float xb_9 = __uint_as_float(xw[9] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_9 * __uint_as_float(ww[9] << 16), xa_9, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_9 * __uint_as_float(ww[9] & 4294901760u), xb_9, dqv[0]);
                    acc[18] = __fmaf_rn(ga_9 * xa_9, rq, acc[18]);
                    acc[19] = __fmaf_rn(gb_9 * xb_9, rq, acc[19]);
                    float ga_10 = __uint_as_float(gw[10] << 16);
                    float gb_10 = __uint_as_float(gw[10] & 4294901760u);
                    float xa_10 = __uint_as_float(xw[10] << 16);
                    float xb_10 = __uint_as_float(xw[10] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_10 * __uint_as_float(ww[10] << 16), xa_10, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_10 * __uint_as_float(ww[10] & 4294901760u), xb_10, dqv[0]);
                    acc[20] = __fmaf_rn(ga_10 * xa_10, rq, acc[20]);
                    acc[21] = __fmaf_rn(gb_10 * xb_10, rq, acc[21]);
                    float ga_11 = __uint_as_float(gw[11] << 16);
                    float gb_11 = __uint_as_float(gw[11] & 4294901760u);
                    float xa_11 = __uint_as_float(xw[11] << 16);
                    float xb_11 = __uint_as_float(xw[11] & 4294901760u);
                    dqv[0] = __fmaf_rn(ga_11 * __uint_as_float(ww[11] << 16), xa_11, dqv[0]);
                    dqv[0] = __fmaf_rn(gb_11 * __uint_as_float(ww[11] & 4294901760u), xb_11, dqv[0]);
                    acc[22] = __fmaf_rn(ga_11 * xa_11, rq, acc[22]);
                    acc[23] = __fmaf_rn(gb_11 * xb_11, rq, acc[23]);
                    float ga_12 = __uint_as_float(gw[12] << 16);
                    float gb_12 = __uint_as_float(gw[12] & 4294901760u);
                    float xa_12 = __uint_as_float(xw[12] << 16);
                    float xb_12 = __uint_as_float(xw[12] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_12 * __uint_as_float(ww[12] << 16), xa_12, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_12 * __uint_as_float(ww[12] & 4294901760u), xb_12, dqv[1]);
                    acc[24] = __fmaf_rn(ga_12 * xa_12, rq, acc[24]);
                    acc[25] = __fmaf_rn(gb_12 * xb_12, rq, acc[25]);
                    float ga_13 = __uint_as_float(gw[13] << 16);
                    float gb_13 = __uint_as_float(gw[13] & 4294901760u);
                    float xa_13 = __uint_as_float(xw[13] << 16);
                    float xb_13 = __uint_as_float(xw[13] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_13 * __uint_as_float(ww[13] << 16), xa_13, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_13 * __uint_as_float(ww[13] & 4294901760u), xb_13, dqv[1]);
                    acc[26] = __fmaf_rn(ga_13 * xa_13, rq, acc[26]);
                    acc[27] = __fmaf_rn(gb_13 * xb_13, rq, acc[27]);
                    float ga_14 = __uint_as_float(gw[14] << 16);
                    float gb_14 = __uint_as_float(gw[14] & 4294901760u);
                    float xa_14 = __uint_as_float(xw[14] << 16);
                    float xb_14 = __uint_as_float(xw[14] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_14 * __uint_as_float(ww[14] << 16), xa_14, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_14 * __uint_as_float(ww[14] & 4294901760u), xb_14, dqv[1]);
                    acc[28] = __fmaf_rn(ga_14 * xa_14, rq, acc[28]);
                    acc[29] = __fmaf_rn(gb_14 * xb_14, rq, acc[29]);
                    float ga_15 = __uint_as_float(gw[15] << 16);
                    float gb_15 = __uint_as_float(gw[15] & 4294901760u);
                    float xa_15 = __uint_as_float(xw[15] << 16);
                    float xb_15 = __uint_as_float(xw[15] & 4294901760u);
                    dqv[1] = __fmaf_rn(ga_15 * __uint_as_float(ww[15] << 16), xa_15, dqv[1]);
                    dqv[1] = __fmaf_rn(gb_15 * __uint_as_float(ww[15] & 4294901760u), xb_15, dqv[1]);
                    acc[30] = __fmaf_rn(ga_15 * xa_15, rq, acc[30]);
                    acc[31] = __fmaf_rn(gb_15 * xb_15, rq, acc[31]);
                    float dsj_0 = dqv[0];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsj_0 += __shfl_xor_sync(0xFFFFFFFF, dsj_0, offset);
                    if (lane == 0) {
                        dots[(k & 1) * 12 + warp] = dsj_0;
                    }
                    float dsj_1 = dqv[1];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        dsj_1 += __shfl_xor_sync(0xFFFFFFFF, dsj_1, offset);
                    if (lane == 0) {
                        dots[(k & 1) * 12 + 6 + warp] = dsj_1;
                    }
                    int par = (k & 1) * 12;
                    asm volatile("barrier.sync 1, 192;" ::: "memory");
                    float tot = dots[par];
                    tot = tot + dots[par + 1];
                    tot = tot + dots[par + 1 + 1];
                    tot = tot + dots[par + 2 + 1];
                    tot = tot + dots[par + 3 + 1];
                    tot = tot + dots[par + 4 + 1];
                    tot = tot + dots[par + 5 + 1];
                    tot = tot + dots[par + 6 + 1];
                    tot = tot + dots[par + 7 + 1];
                    tot = tot + dots[par + 8 + 1];
                    tot = tot + dots[par + 9 + 1];
                    tot = tot + dots[par + 10 + 1];
                    float cq = __fdiv_rn(tot, 6144.0f);
                    float kx = rq * rq * rq * cq;
                    float nkx = -kx;
                    unsigned long long dx_base = (unsigned long long)row * 6144ull;
                    __int128_t _smem_raw_12;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_12) : "r"(sbase + (0 + t) * 8 * 2));
                    gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[0];
                    gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[1];
                    gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[2];
                    gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_12)[3];
                    __int128_t _smem_raw_13;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_13) : "r"(sbase + 12288 + (0 + t) * 8 * 2));
                    xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[0];
                    xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[1];
                    xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[2];
                    xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_13)[3];
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
                    __int128_t _smem_raw_14;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_14) : "r"(sbase + (192 + t) * 8 * 2));
                    gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[0];
                    gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[1];
                    gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[2];
                    gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_14)[3];
                    __int128_t _smem_raw_15;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_15) : "r"(sbase + 12288 + (192 + t) * 8 * 2));
                    xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[0];
                    xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[1];
                    xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[2];
                    xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_15)[3];
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
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((192 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    __int128_t _smem_raw_16;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_16) : "r"(sbase + (384 + t) * 8 * 2));
                    gw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[0];
                    gw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[1];
                    gw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[2];
                    gw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_16)[3];
                    __int128_t _smem_raw_17;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_17) : "r"(sbase + 12288 + (384 + t) * 8 * 2));
                    xw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[0];
                    xw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[1];
                    xw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[2];
                    xw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_17)[3];
                    float out_2[8];
                    float g0_8 = __uint_as_float(gw[8] << 16);
                    float g1_8 = __uint_as_float(gw[8] & 4294901760u);
                    float x0_8 = __uint_as_float(xw[8] << 16);
                    float x1_8 = __uint_as_float(xw[8] & 4294901760u);
                    out_2[0] = __fmaf_rn(nkx, x0_8, rq * (g0_8 * __uint_as_float(ww[8] << 16)));
                    out_2[1] = __fmaf_rn(nkx, x1_8, rq * (g1_8 * __uint_as_float(ww[8] & 4294901760u)));
                    float g0_9 = __uint_as_float(gw[9] << 16);
                    float g1_9 = __uint_as_float(gw[9] & 4294901760u);
                    float x0_9 = __uint_as_float(xw[9] << 16);
                    float x1_9 = __uint_as_float(xw[9] & 4294901760u);
                    out_2[2] = __fmaf_rn(nkx, x0_9, rq * (g0_9 * __uint_as_float(ww[9] << 16)));
                    out_2[3] = __fmaf_rn(nkx, x1_9, rq * (g1_9 * __uint_as_float(ww[9] & 4294901760u)));
                    float g0_10 = __uint_as_float(gw[10] << 16);
                    float g1_10 = __uint_as_float(gw[10] & 4294901760u);
                    float x0_10 = __uint_as_float(xw[10] << 16);
                    float x1_10 = __uint_as_float(xw[10] & 4294901760u);
                    out_2[4] = __fmaf_rn(nkx, x0_10, rq * (g0_10 * __uint_as_float(ww[10] << 16)));
                    out_2[5] = __fmaf_rn(nkx, x1_10, rq * (g1_10 * __uint_as_float(ww[10] & 4294901760u)));
                    float g0_11 = __uint_as_float(gw[11] << 16);
                    float g1_11 = __uint_as_float(gw[11] & 4294901760u);
                    float x0_11 = __uint_as_float(xw[11] << 16);
                    float x1_11 = __uint_as_float(xw[11] & 4294901760u);
                    out_2[6] = __fmaf_rn(nkx, x0_11, rq * (g0_11 * __uint_as_float(ww[11] << 16)));
                    out_2[7] = __fmaf_rn(nkx, x1_11, rq * (g1_11 * __uint_as_float(ww[11] & 4294901760u)));
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((384 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
                    __int128_t _smem_raw_18;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(sbase + (576 + t) * 8 * 2));
                    gw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
                    gw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
                    gw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
                    gw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
                    __int128_t _smem_raw_19;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(sbase + 12288 + (576 + t) * 8 * 2));
                    xw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                    xw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                    xw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
                    xw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
                    float out_3[8];
                    float g0_12 = __uint_as_float(gw[12] << 16);
                    float g1_12 = __uint_as_float(gw[12] & 4294901760u);
                    float x0_12 = __uint_as_float(xw[12] << 16);
                    float x1_12 = __uint_as_float(xw[12] & 4294901760u);
                    out_3[0] = __fmaf_rn(nkx, x0_12, rq * (g0_12 * __uint_as_float(ww[12] << 16)));
                    out_3[1] = __fmaf_rn(nkx, x1_12, rq * (g1_12 * __uint_as_float(ww[12] & 4294901760u)));
                    float g0_13 = __uint_as_float(gw[13] << 16);
                    float g1_13 = __uint_as_float(gw[13] & 4294901760u);
                    float x0_13 = __uint_as_float(xw[13] << 16);
                    float x1_13 = __uint_as_float(xw[13] & 4294901760u);
                    out_3[2] = __fmaf_rn(nkx, x0_13, rq * (g0_13 * __uint_as_float(ww[13] << 16)));
                    out_3[3] = __fmaf_rn(nkx, x1_13, rq * (g1_13 * __uint_as_float(ww[13] & 4294901760u)));
                    float g0_14 = __uint_as_float(gw[14] << 16);
                    float g1_14 = __uint_as_float(gw[14] & 4294901760u);
                    float x0_14 = __uint_as_float(xw[14] << 16);
                    float x1_14 = __uint_as_float(xw[14] & 4294901760u);
                    out_3[4] = __fmaf_rn(nkx, x0_14, rq * (g0_14 * __uint_as_float(ww[14] << 16)));
                    out_3[5] = __fmaf_rn(nkx, x1_14, rq * (g1_14 * __uint_as_float(ww[14] & 4294901760u)));
                    float g0_15 = __uint_as_float(gw[15] << 16);
                    float g1_15 = __uint_as_float(gw[15] & 4294901760u);
                    float x0_15 = __uint_as_float(xw[15] << 16);
                    float x1_15 = __uint_as_float(xw[15] & 4294901760u);
                    out_3[6] = __fmaf_rn(nkx, x0_15, rq * (g0_15 * __uint_as_float(ww[15] << 16)));
                    out_3[7] = __fmaf_rn(nkx, x1_15, rq * (g1_15 * __uint_as_float(ww[15] & 4294901760u)));
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)((576 + t) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
                    mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 3)) * 8);
                    row = row + n_chunks;
                    k += 1;
                }
                unsigned long long pbase = (unsigned long long)slot * 6144ull;
                {
                    unsigned _stv8_20_0 = __float_as_uint(acc[0 + 0]);
                    unsigned _stv8_20_1 = __float_as_uint(acc[0 + 1]);
                    unsigned _stv8_20_2 = __float_as_uint(acc[0 + 2]);
                    unsigned _stv8_20_3 = __float_as_uint(acc[0 + 3]);
                    unsigned _stv8_20_4 = __float_as_uint(acc[0 + 4]);
                    unsigned _stv8_20_5 = __float_as_uint(acc[0 + 5]);
                    unsigned _stv8_20_6 = __float_as_uint(acc[0 + 6]);
                    unsigned _stv8_20_7 = __float_as_uint(acc[0 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((0 + t) * 8)))), "r"(_stv8_20_0), "r"(_stv8_20_1), "r"(_stv8_20_2), "r"(_stv8_20_3), "r"(_stv8_20_4), "r"(_stv8_20_5), "r"(_stv8_20_6), "r"(_stv8_20_7) : "memory");
                }
                {
                    unsigned _stv8_21_0 = __float_as_uint(acc[8 + 0]);
                    unsigned _stv8_21_1 = __float_as_uint(acc[8 + 1]);
                    unsigned _stv8_21_2 = __float_as_uint(acc[8 + 2]);
                    unsigned _stv8_21_3 = __float_as_uint(acc[8 + 3]);
                    unsigned _stv8_21_4 = __float_as_uint(acc[8 + 4]);
                    unsigned _stv8_21_5 = __float_as_uint(acc[8 + 5]);
                    unsigned _stv8_21_6 = __float_as_uint(acc[8 + 6]);
                    unsigned _stv8_21_7 = __float_as_uint(acc[8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((192 + t) * 8)))), "r"(_stv8_21_0), "r"(_stv8_21_1), "r"(_stv8_21_2), "r"(_stv8_21_3), "r"(_stv8_21_4), "r"(_stv8_21_5), "r"(_stv8_21_6), "r"(_stv8_21_7) : "memory");
                }
                {
                    unsigned _stv8_22_0 = __float_as_uint(acc[16 + 0]);
                    unsigned _stv8_22_1 = __float_as_uint(acc[16 + 1]);
                    unsigned _stv8_22_2 = __float_as_uint(acc[16 + 2]);
                    unsigned _stv8_22_3 = __float_as_uint(acc[16 + 3]);
                    unsigned _stv8_22_4 = __float_as_uint(acc[16 + 4]);
                    unsigned _stv8_22_5 = __float_as_uint(acc[16 + 5]);
                    unsigned _stv8_22_6 = __float_as_uint(acc[16 + 6]);
                    unsigned _stv8_22_7 = __float_as_uint(acc[16 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((384 + t) * 8)))), "r"(_stv8_22_0), "r"(_stv8_22_1), "r"(_stv8_22_2), "r"(_stv8_22_3), "r"(_stv8_22_4), "r"(_stv8_22_5), "r"(_stv8_22_6), "r"(_stv8_22_7) : "memory");
                }
                {
                    unsigned _stv8_23_0 = __float_as_uint(acc[24 + 0]);
                    unsigned _stv8_23_1 = __float_as_uint(acc[24 + 1]);
                    unsigned _stv8_23_2 = __float_as_uint(acc[24 + 2]);
                    unsigned _stv8_23_3 = __float_as_uint(acc[24 + 3]);
                    unsigned _stv8_23_4 = __float_as_uint(acc[24 + 4]);
                    unsigned _stv8_23_5 = __float_as_uint(acc[24 + 5]);
                    unsigned _stv8_23_6 = __float_as_uint(acc[24 + 6]);
                    unsigned _stv8_23_7 = __float_as_uint(acc[24 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(partial + (pbase + (unsigned long long)((576 + t) * 8)))), "r"(_stv8_23_0), "r"(_stv8_23_1), "r"(_stv8_23_2), "r"(_stv8_23_3), "r"(_stv8_23_4), "r"(_stv8_23_5), "r"(_stv8_23_6), "r"(_stv8_23_7) : "memory");
                }
                slot = slot + (int)num_bids;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: prod ----
    if (warp == 6) {
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
                    int _stage_empty = (int)((unsigned int)pk % 3);
                    int _parity_empty = (int)((unsigned int)pk / 3 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 3)) * 8, 24576);
                    int sb = p_ring + pk % 3 * 24576;
                    long long src = (long long)p_row;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sb), "l"(reinterpret_cast<const uint8_t*>(g + src * g_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 3)) * 8),
                           "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sb + 12288), "l"(reinterpret_cast<const uint8_t*>(x + src * x_stride_p0)), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)pk % 3)) * 8),
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
