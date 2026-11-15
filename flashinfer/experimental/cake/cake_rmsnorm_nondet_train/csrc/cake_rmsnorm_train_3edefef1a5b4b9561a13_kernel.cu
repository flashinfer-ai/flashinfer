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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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



extern "C" {

__global__ __launch_bounds__(416, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_3edefef1a5b4b9561a13(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial));
    __builtin_assume(__isGlobal(partial));

    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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
            float comp[16];
            acc[0] = 0.0f;
            comp[0] = 0.0f;
            acc[1] = 0.0f;
            comp[1] = 0.0f;
            acc[2] = 0.0f;
            comp[2] = 0.0f;
            acc[3] = 0.0f;
            comp[3] = 0.0f;
            acc[4] = 0.0f;
            comp[4] = 0.0f;
            acc[5] = 0.0f;
            comp[5] = 0.0f;
            acc[6] = 0.0f;
            comp[6] = 0.0f;
            acc[7] = 0.0f;
            comp[7] = 0.0f;
            acc[8] = 0.0f;
            comp[8] = 0.0f;
            acc[9] = 0.0f;
            comp[9] = 0.0f;
            acc[10] = 0.0f;
            comp[10] = 0.0f;
            acc[11] = 0.0f;
            comp[11] = 0.0f;
            acc[12] = 0.0f;
            comp[12] = 0.0f;
            acc[13] = 0.0f;
            comp[13] = 0.0f;
            acc[14] = 0.0f;
            comp[14] = 0.0f;
            acc[15] = 0.0f;
            comp[15] = 0.0f;
            int c = 0;
            int first_c = 0;
            int rows_mine = 0;
            int slot = 0;
            int ring_at = (int)ring_addr;
            unsigned int gw[8];
            unsigned int xw[8];
            unsigned int hw[1];
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
                rows_mine = rows_mine + nval;
                if (k == 0) {
                    if (tid == 0) {
                        unsigned int _atomic_fetch_add_2 = atomicAdd(&counters[3], 1);
                        slot = (int)_atomic_fetch_add_2;
                    }
                }
                int rbase = ring_at + k % 4 * 24576 + rg * 12288;
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(rbase + tir * 8 * 2));
                gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(rbase + 12288 + tir * 8 * 2));
                xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
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
                float _warp_reduce_7 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_7 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_7, offset);
                dot[0] = _warp_reduce_7;
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
                float2 rr2 = make_float2(rq, rq);
                if (row0 + rg < row0 + nval) {
                    float out_0[8];
                    float g0_0 = __uint_as_float(gw[0] << 16);
                    float g1_0 = __uint_as_float(gw[0] & 4294901760u);
                    float x0_0 = __uint_as_float(xw[0] << 16);
                    float x1_0 = __uint_as_float(xw[0] & 4294901760u);
                    float2 _f32x2_mul_a_8 = make_float2(g0_0, g1_0);
                    float2 _f32x2_mul_b_9 = make_float2(x0_0, x1_0);
                    float2 gx;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                    float2 p;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rr2));
                    float2 a = make_float2(acc[0], acc[1]);
                    float2 s = add_f32x2_noftz(a, p);
                    float2 bn = sub_f32x2_noftz(a, s);
                    float2 av = add_f32x2_noftz(s, bn);
                    float2 db = fma_f32x2_rn_noftz(gx, rr2, bn);
                    float2 c_0 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[0], comp[1]), sub_f32x2_noftz(a, av)), db);
                    acc[0] = s.x;
                    acc[1] = s.y;
                    comp[0] = c_0.x;
                    comp[1] = c_0.y;
                    float y0_0 = __fmaf_rn(nkx, x0_0, rq * (g0_0 * __uint_as_float(ww[0] << 16)));
                    float y1_0 = __fmaf_rn(nkx, x1_0, rq * (g1_0 * __uint_as_float(ww[0] & 4294901760u)));
                    out_0[0] = y0_0;
                    out_0[1] = y1_0;
                    float g0_1 = __uint_as_float(gw[1] << 16);
                    float g1_1 = __uint_as_float(gw[1] & 4294901760u);
                    float x0_1 = __uint_as_float(xw[1] << 16);
                    float x1_1 = __uint_as_float(xw[1] & 4294901760u);
                    float2 _f32x2_mul_a_10 = make_float2(g0_1, g1_1);
                    float2 _f32x2_mul_b_11 = make_float2(x0_1, x1_1);
                    float2 gx_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                    float2 p_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_0) : "l"(*(const unsigned long long*)&gx_0), "l"(*(const unsigned long long*)&rr2));
                    float2 a_0 = make_float2(acc[2], acc[3]);
                    float2 s_0 = add_f32x2_noftz(a_0, p_0);
                    float2 bn_0 = sub_f32x2_noftz(a_0, s_0);
                    float2 av_0 = add_f32x2_noftz(s_0, bn_0);
                    float2 db_0 = fma_f32x2_rn_noftz(gx_0, rr2, bn_0);
                    float2 c_1 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[2], comp[3]), sub_f32x2_noftz(a_0, av_0)), db_0);
                    acc[2] = s_0.x;
                    acc[3] = s_0.y;
                    comp[2] = c_1.x;
                    comp[3] = c_1.y;
                    float y0_1 = __fmaf_rn(nkx, x0_1, rq * (g0_1 * __uint_as_float(ww[1] << 16)));
                    float y1_1 = __fmaf_rn(nkx, x1_1, rq * (g1_1 * __uint_as_float(ww[1] & 4294901760u)));
                    out_0[2] = y0_1;
                    out_0[3] = y1_1;
                    float g0_2 = __uint_as_float(gw[2] << 16);
                    float g1_2 = __uint_as_float(gw[2] & 4294901760u);
                    float x0_2 = __uint_as_float(xw[2] << 16);
                    float x1_2 = __uint_as_float(xw[2] & 4294901760u);
                    float2 _f32x2_mul_a_12 = make_float2(g0_2, g1_2);
                    float2 _f32x2_mul_b_13 = make_float2(x0_2, x1_2);
                    float2 gx_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_13));
                    float2 p_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_1) : "l"(*(const unsigned long long*)&gx_1), "l"(*(const unsigned long long*)&rr2));
                    float2 a_1 = make_float2(acc[4], acc[5]);
                    float2 s_1 = add_f32x2_noftz(a_1, p_1);
                    float2 bn_1 = sub_f32x2_noftz(a_1, s_1);
                    float2 av_1 = add_f32x2_noftz(s_1, bn_1);
                    float2 db_1 = fma_f32x2_rn_noftz(gx_1, rr2, bn_1);
                    float2 c_2 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[4], comp[5]), sub_f32x2_noftz(a_1, av_1)), db_1);
                    acc[4] = s_1.x;
                    acc[5] = s_1.y;
                    comp[4] = c_2.x;
                    comp[5] = c_2.y;
                    float y0_2 = __fmaf_rn(nkx, x0_2, rq * (g0_2 * __uint_as_float(ww[2] << 16)));
                    float y1_2 = __fmaf_rn(nkx, x1_2, rq * (g1_2 * __uint_as_float(ww[2] & 4294901760u)));
                    out_0[4] = y0_2;
                    out_0[5] = y1_2;
                    float g0_3 = __uint_as_float(gw[3] << 16);
                    float g1_3 = __uint_as_float(gw[3] & 4294901760u);
                    float x0_3 = __uint_as_float(xw[3] << 16);
                    float x1_3 = __uint_as_float(xw[3] & 4294901760u);
                    float2 _f32x2_mul_a_14 = make_float2(g0_3, g1_3);
                    float2 _f32x2_mul_b_15 = make_float2(x0_3, x1_3);
                    float2 gx_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_15));
                    float2 p_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_2) : "l"(*(const unsigned long long*)&gx_2), "l"(*(const unsigned long long*)&rr2));
                    float2 a_2 = make_float2(acc[6], acc[7]);
                    float2 s_2 = add_f32x2_noftz(a_2, p_2);
                    float2 bn_2 = sub_f32x2_noftz(a_2, s_2);
                    float2 av_2 = add_f32x2_noftz(s_2, bn_2);
                    float2 db_2 = fma_f32x2_rn_noftz(gx_2, rr2, bn_2);
                    float2 c_3 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[6], comp[7]), sub_f32x2_noftz(a_2, av_2)), db_2);
                    acc[6] = s_2.x;
                    acc[7] = s_2.y;
                    comp[6] = c_3.x;
                    comp[7] = c_3.y;
                    float y0_3 = __fmaf_rn(nkx, x0_3, rq * (g0_3 * __uint_as_float(ww[3] << 16)));
                    float y1_3 = __fmaf_rn(nkx, x1_3, rq * (g1_3 * __uint_as_float(ww[3] & 4294901760u)));
                    out_0[6] = y0_3;
                    out_0[7] = y1_3;
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
                    float2 _f32x2_mul_a_16 = make_float2(g0_4, g1_4);
                    float2 _f32x2_mul_b_17 = make_float2(x0_4, x1_4);
                    float2 gx_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                    float2 p_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_3) : "l"(*(const unsigned long long*)&gx_3), "l"(*(const unsigned long long*)&rr2));
                    float2 a_3 = make_float2(acc[8], acc[9]);
                    float2 s_3 = add_f32x2_noftz(a_3, p_3);
                    float2 bn_3 = sub_f32x2_noftz(a_3, s_3);
                    float2 av_3 = add_f32x2_noftz(s_3, bn_3);
                    float2 db_3 = fma_f32x2_rn_noftz(gx_3, rr2, bn_3);
                    float2 c_4 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[8], comp[9]), sub_f32x2_noftz(a_3, av_3)), db_3);
                    acc[8] = s_3.x;
                    acc[9] = s_3.y;
                    comp[8] = c_4.x;
                    comp[9] = c_4.y;
                    float y0_4 = __fmaf_rn(nkx, x0_4, rq * (g0_4 * __uint_as_float(ww[4] << 16)));
                    float y1_4 = __fmaf_rn(nkx, x1_4, rq * (g1_4 * __uint_as_float(ww[4] & 4294901760u)));
                    out_1[0] = y0_4;
                    out_1[1] = y1_4;
                    float g0_5 = __uint_as_float(gw[5] << 16);
                    float g1_5 = __uint_as_float(gw[5] & 4294901760u);
                    float x0_5 = __uint_as_float(xw[5] << 16);
                    float x1_5 = __uint_as_float(xw[5] & 4294901760u);
                    float2 _f32x2_mul_a_18 = make_float2(g0_5, g1_5);
                    float2 _f32x2_mul_b_19 = make_float2(x0_5, x1_5);
                    float2 gx_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                    float2 p_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_4) : "l"(*(const unsigned long long*)&gx_4), "l"(*(const unsigned long long*)&rr2));
                    float2 a_4 = make_float2(acc[10], acc[11]);
                    float2 s_4 = add_f32x2_noftz(a_4, p_4);
                    float2 bn_4 = sub_f32x2_noftz(a_4, s_4);
                    float2 av_4 = add_f32x2_noftz(s_4, bn_4);
                    float2 db_4 = fma_f32x2_rn_noftz(gx_4, rr2, bn_4);
                    float2 c_5 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[10], comp[11]), sub_f32x2_noftz(a_4, av_4)), db_4);
                    acc[10] = s_4.x;
                    acc[11] = s_4.y;
                    comp[10] = c_5.x;
                    comp[11] = c_5.y;
                    float y0_5 = __fmaf_rn(nkx, x0_5, rq * (g0_5 * __uint_as_float(ww[5] << 16)));
                    float y1_5 = __fmaf_rn(nkx, x1_5, rq * (g1_5 * __uint_as_float(ww[5] & 4294901760u)));
                    out_1[2] = y0_5;
                    out_1[3] = y1_5;
                    float g0_6 = __uint_as_float(gw[6] << 16);
                    float g1_6 = __uint_as_float(gw[6] & 4294901760u);
                    float x0_6 = __uint_as_float(xw[6] << 16);
                    float x1_6 = __uint_as_float(xw[6] & 4294901760u);
                    float2 _f32x2_mul_a_20 = make_float2(g0_6, g1_6);
                    float2 _f32x2_mul_b_21 = make_float2(x0_6, x1_6);
                    float2 gx_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_21));
                    float2 p_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_5) : "l"(*(const unsigned long long*)&gx_5), "l"(*(const unsigned long long*)&rr2));
                    float2 a_5 = make_float2(acc[12], acc[13]);
                    float2 s_5 = add_f32x2_noftz(a_5, p_5);
                    float2 bn_5 = sub_f32x2_noftz(a_5, s_5);
                    float2 av_5 = add_f32x2_noftz(s_5, bn_5);
                    float2 db_5 = fma_f32x2_rn_noftz(gx_5, rr2, bn_5);
                    float2 c_6 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[12], comp[13]), sub_f32x2_noftz(a_5, av_5)), db_5);
                    acc[12] = s_5.x;
                    acc[13] = s_5.y;
                    comp[12] = c_6.x;
                    comp[13] = c_6.y;
                    float y0_6 = __fmaf_rn(nkx, x0_6, rq * (g0_6 * __uint_as_float(ww[6] << 16)));
                    float y1_6 = __fmaf_rn(nkx, x1_6, rq * (g1_6 * __uint_as_float(ww[6] & 4294901760u)));
                    out_1[4] = y0_6;
                    out_1[5] = y1_6;
                    float g0_7 = __uint_as_float(gw[7] << 16);
                    float g1_7 = __uint_as_float(gw[7] & 4294901760u);
                    float x0_7 = __uint_as_float(xw[7] << 16);
                    float x1_7 = __uint_as_float(xw[7] & 4294901760u);
                    float2 _f32x2_mul_a_22 = make_float2(g0_7, g1_7);
                    float2 _f32x2_mul_b_23 = make_float2(x0_7, x1_7);
                    float2 gx_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_23));
                    float2 p_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_6) : "l"(*(const unsigned long long*)&gx_6), "l"(*(const unsigned long long*)&rr2));
                    float2 a_6 = make_float2(acc[14], acc[15]);
                    float2 s_6 = add_f32x2_noftz(a_6, p_6);
                    float2 bn_6 = sub_f32x2_noftz(a_6, s_6);
                    float2 av_6 = add_f32x2_noftz(s_6, bn_6);
                    float2 db_6 = fma_f32x2_rn_noftz(gx_6, rr2, bn_6);
                    float2 c_7 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[14], comp[15]), sub_f32x2_noftz(a_6, av_6)), db_6);
                    acc[14] = s_6.x;
                    acc[15] = s_6.y;
                    comp[14] = c_7.x;
                    comp[15] = c_7.y;
                    float y0_7 = __fmaf_rn(nkx, x0_7, rq * (g0_7 * __uint_as_float(ww[7] << 16)));
                    float y1_7 = __fmaf_rn(nkx, x1_7, rq * (g1_7 * __uint_as_float(ww[7] & 4294901760u)));
                    out_1[6] = y0_7;
                    out_1[7] = y1_7;
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_1[_lp*2 + 0], out_1[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144ull + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
                }
            }
            pairs[2 * (rg * 6144 + tir)] = acc[0];
            pairs[2 * (rg * 6144 + tir) + 1] = comp[0];
            pairs[2 * (rg * 6144 + 384 + tir)] = acc[1];
            pairs[2 * (rg * 6144 + 384 + tir) + 1] = comp[1];
            pairs[2 * (rg * 6144 + 768 + tir)] = acc[2];
            pairs[2 * (rg * 6144 + 768 + tir) + 1] = comp[2];
            pairs[2 * (rg * 6144 + 1152 + tir)] = acc[3];
            pairs[2 * (rg * 6144 + 1152 + tir) + 1] = comp[3];
            pairs[2 * (rg * 6144 + 1536 + tir)] = acc[4];
            pairs[2 * (rg * 6144 + 1536 + tir) + 1] = comp[4];
            pairs[2 * (rg * 6144 + 1920 + tir)] = acc[5];
            pairs[2 * (rg * 6144 + 1920 + tir) + 1] = comp[5];
            pairs[2 * (rg * 6144 + 2304 + tir)] = acc[6];
            pairs[2 * (rg * 6144 + 2304 + tir) + 1] = comp[6];
            pairs[2 * (rg * 6144 + 2688 + tir)] = acc[7];
            pairs[2 * (rg * 6144 + 2688 + tir) + 1] = comp[7];
            pairs[2 * (rg * 6144 + 3072 + tir)] = acc[8];
            pairs[2 * (rg * 6144 + 3072 + tir) + 1] = comp[8];
            pairs[2 * (rg * 6144 + 3456 + tir)] = acc[9];
            pairs[2 * (rg * 6144 + 3456 + tir) + 1] = comp[9];
            pairs[2 * (rg * 6144 + 3840 + tir)] = acc[10];
            pairs[2 * (rg * 6144 + 3840 + tir) + 1] = comp[10];
            pairs[2 * (rg * 6144 + 4224 + tir)] = acc[11];
            pairs[2 * (rg * 6144 + 4224 + tir) + 1] = comp[11];
            pairs[2 * (rg * 6144 + 4608 + tir)] = acc[12];
            pairs[2 * (rg * 6144 + 4608 + tir) + 1] = comp[12];
            pairs[2 * (rg * 6144 + 4992 + tir)] = acc[13];
            pairs[2 * (rg * 6144 + 4992 + tir) + 1] = comp[13];
            pairs[2 * (rg * 6144 + 5376 + tir)] = acc[14];
            pairs[2 * (rg * 6144 + 5376 + tir) + 1] = comp[14];
            pairs[2 * (rg * 6144 + 5760 + tir)] = acc[15];
            pairs[2 * (rg * 6144 + 5760 + tir) + 1] = comp[15];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            if (tid == 0) {
                if (first_c < n_chunks) {
                    {
                        void* _cpbulk_dst_24 = reinterpret_cast<void*>(partial + (long long)(slot * 12288));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_24), "r"(pairs_addr), "r"((uint32_t)(49152))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (1))), "r"(static_cast<unsigned int>((unsigned int)rows_mine)) : "memory");
                }
                int hs = -1;
                if (c - n_chunks < 128) {
                    hs = c - n_chunks;
                }
                flags[3] = hs;
                if (c == n_chunks + (int)gridDim.x - 1) {
                    *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
                }
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            int hs_c = flags[3];
            if (hs_c >= 0) {
                if (tid == 0) {
                    {
                    unsigned int _acquire_observed;
                    do {
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(counters) + (1))) : "memory");
                    } while (_acquire_observed < static_cast<unsigned int>((unsigned int)T));
                    }
                }
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                asm volatile("fence.acquire.gpu;" ::: "memory");
                unsigned int nw[1];
                {
                    uint32_t _scalar_bits_25;
                    asm volatile("ld.global.cg.b32 %0, [%1];"
                        : "=r"(_scalar_bits_25) : "l"((const void*)(counters + (3))) : "memory");
                    nw[0] = (unsigned int)_scalar_bits_25;
                }
                int grid = (int)num_bids;
                int helpers = ((grid) < (128) ? (grid) : (128));
                int cl = tid % 24;
                int rl = tid / 24;
                #pragma unroll 1
                for (int kk = 0; kk < 128; kk++) {
                    int sl = hs_c + kk * helpers;
                    if (sl >= 128) {
                        break;
                    }
                    int at0 = sl * 48 + 2 * cl;
                    float p_7[4];
                    p_7[0] = 0.0f;
                    p_7[1] = 0.0f;
                    p_7[2] = 0.0f;
                    p_7[3] = 0.0f;
                    float buf[32];
                    #pragma unroll 1
                    for (int s0 = rl; s0 < grid; s0 += 128) {
                        if (s0 < grid) {
                            {
                                unsigned _v4_26_0;
                                unsigned _v4_26_1;
                                unsigned _v4_26_2;
                                unsigned _v4_26_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_26_0), "=r"(_v4_26_1), "=r"(_v4_26_2), "=r"(_v4_26_3) : "l"((const void*)(partial + (s0 * 12288 + 2 * at0))) : "memory");
                                buf[0 + 0] = __uint_as_float(_v4_26_0);
                                buf[0 + 1] = __uint_as_float(_v4_26_1);
                                buf[0 + 2] = __uint_as_float(_v4_26_2);
                                buf[0 + 3] = __uint_as_float(_v4_26_3);
                            }
                        }
                        if (s0 + 16 < grid) {
                            {
                                unsigned _v4_27_0;
                                unsigned _v4_27_1;
                                unsigned _v4_27_2;
                                unsigned _v4_27_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_27_0), "=r"(_v4_27_1), "=r"(_v4_27_2), "=r"(_v4_27_3) : "l"((const void*)(partial + ((s0 + 16) * 12288 + 2 * at0))) : "memory");
                                buf[4 + 0] = __uint_as_float(_v4_27_0);
                                buf[4 + 1] = __uint_as_float(_v4_27_1);
                                buf[4 + 2] = __uint_as_float(_v4_27_2);
                                buf[4 + 3] = __uint_as_float(_v4_27_3);
                            }
                        }
                        if (s0 + 32 < grid) {
                            {
                                unsigned _v4_28_0;
                                unsigned _v4_28_1;
                                unsigned _v4_28_2;
                                unsigned _v4_28_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_28_0), "=r"(_v4_28_1), "=r"(_v4_28_2), "=r"(_v4_28_3) : "l"((const void*)(partial + ((s0 + 32) * 12288 + 2 * at0))) : "memory");
                                buf[8 + 0] = __uint_as_float(_v4_28_0);
                                buf[8 + 1] = __uint_as_float(_v4_28_1);
                                buf[8 + 2] = __uint_as_float(_v4_28_2);
                                buf[8 + 3] = __uint_as_float(_v4_28_3);
                            }
                        }
                        if (s0 + 48 < grid) {
                            {
                                unsigned _v4_29_0;
                                unsigned _v4_29_1;
                                unsigned _v4_29_2;
                                unsigned _v4_29_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_29_0), "=r"(_v4_29_1), "=r"(_v4_29_2), "=r"(_v4_29_3) : "l"((const void*)(partial + ((s0 + 48) * 12288 + 2 * at0))) : "memory");
                                buf[12 + 0] = __uint_as_float(_v4_29_0);
                                buf[12 + 1] = __uint_as_float(_v4_29_1);
                                buf[12 + 2] = __uint_as_float(_v4_29_2);
                                buf[12 + 3] = __uint_as_float(_v4_29_3);
                            }
                        }
                        if (s0 + 64 < grid) {
                            {
                                unsigned _v4_30_0;
                                unsigned _v4_30_1;
                                unsigned _v4_30_2;
                                unsigned _v4_30_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_30_0), "=r"(_v4_30_1), "=r"(_v4_30_2), "=r"(_v4_30_3) : "l"((const void*)(partial + ((s0 + 64) * 12288 + 2 * at0))) : "memory");
                                buf[16 + 0] = __uint_as_float(_v4_30_0);
                                buf[16 + 1] = __uint_as_float(_v4_30_1);
                                buf[16 + 2] = __uint_as_float(_v4_30_2);
                                buf[16 + 3] = __uint_as_float(_v4_30_3);
                            }
                        }
                        if (s0 + 80 < grid) {
                            {
                                unsigned _v4_31_0;
                                unsigned _v4_31_1;
                                unsigned _v4_31_2;
                                unsigned _v4_31_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_31_0), "=r"(_v4_31_1), "=r"(_v4_31_2), "=r"(_v4_31_3) : "l"((const void*)(partial + ((s0 + 80) * 12288 + 2 * at0))) : "memory");
                                buf[20 + 0] = __uint_as_float(_v4_31_0);
                                buf[20 + 1] = __uint_as_float(_v4_31_1);
                                buf[20 + 2] = __uint_as_float(_v4_31_2);
                                buf[20 + 3] = __uint_as_float(_v4_31_3);
                            }
                        }
                        if (s0 + 96 < grid) {
                            {
                                unsigned _v4_32_0;
                                unsigned _v4_32_1;
                                unsigned _v4_32_2;
                                unsigned _v4_32_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_32_0), "=r"(_v4_32_1), "=r"(_v4_32_2), "=r"(_v4_32_3) : "l"((const void*)(partial + ((s0 + 96) * 12288 + 2 * at0))) : "memory");
                                buf[24 + 0] = __uint_as_float(_v4_32_0);
                                buf[24 + 1] = __uint_as_float(_v4_32_1);
                                buf[24 + 2] = __uint_as_float(_v4_32_2);
                                buf[24 + 3] = __uint_as_float(_v4_32_3);
                            }
                        }
                        if (s0 + 112 < grid) {
                            {
                                unsigned _v4_33_0;
                                unsigned _v4_33_1;
                                unsigned _v4_33_2;
                                unsigned _v4_33_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_33_0), "=r"(_v4_33_1), "=r"(_v4_33_2), "=r"(_v4_33_3) : "l"((const void*)(partial + ((s0 + 112) * 12288 + 2 * at0))) : "memory");
                                buf[28 + 0] = __uint_as_float(_v4_33_0);
                                buf[28 + 1] = __uint_as_float(_v4_33_1);
                                buf[28 + 2] = __uint_as_float(_v4_33_2);
                                buf[28 + 3] = __uint_as_float(_v4_33_3);
                            }
                        }
                        if (s0 < (int)nw[0]) {
                            float s_7 = p_7[0] + buf[0];
                            float bb = s_7 - p_7[0];
                            float err = p_7[0] - (s_7 - bb) + (buf[0] - bb);
                            p_7[1] = p_7[1] + buf[1] + err;
                            p_7[0] = s_7;
                            float s_7_1 = p_7[2] + buf[2];
                            float bb_0 = s_7_1 - p_7[2];
                            float err_0 = p_7[2] - (s_7_1 - bb_0) + (buf[2] - bb_0);
                            p_7[3] = p_7[3] + buf[3] + err_0;
                            p_7[2] = s_7_1;
                        }
                        if (s0 + 16 < (int)nw[0]) {
                            float s_8 = p_7[0] + buf[4];
                            float bb_1 = s_8 - p_7[0];
                            float err_1 = p_7[0] - (s_8 - bb_1) + (buf[4] - bb_1);
                            p_7[1] = p_7[1] + buf[5] + err_1;
                            p_7[0] = s_8;
                            float s_8_1 = p_7[2] + buf[6];
                            float bb_1_1 = s_8_1 - p_7[2];
                            float err_1_1 = p_7[2] - (s_8_1 - bb_1_1) + (buf[6] - bb_1_1);
                            p_7[3] = p_7[3] + buf[7] + err_1_1;
                            p_7[2] = s_8_1;
                        }
                        if (s0 + 32 < (int)nw[0]) {
                            float s_9 = p_7[0] + buf[8];
                            float bb_2 = s_9 - p_7[0];
                            float err_2 = p_7[0] - (s_9 - bb_2) + (buf[8] - bb_2);
                            p_7[1] = p_7[1] + buf[9] + err_2;
                            p_7[0] = s_9;
                            float s_9_1 = p_7[2] + buf[10];
                            float bb_2_1 = s_9_1 - p_7[2];
                            float err_2_1 = p_7[2] - (s_9_1 - bb_2_1) + (buf[10] - bb_2_1);
                            p_7[3] = p_7[3] + buf[11] + err_2_1;
                            p_7[2] = s_9_1;
                        }
                        if (s0 + 48 < (int)nw[0]) {
                            float s_10 = p_7[0] + buf[12];
                            float bb_3 = s_10 - p_7[0];
                            float err_3 = p_7[0] - (s_10 - bb_3) + (buf[12] - bb_3);
                            p_7[1] = p_7[1] + buf[13] + err_3;
                            p_7[0] = s_10;
                            float s_10_1 = p_7[2] + buf[14];
                            float bb_3_1 = s_10_1 - p_7[2];
                            float err_3_1 = p_7[2] - (s_10_1 - bb_3_1) + (buf[14] - bb_3_1);
                            p_7[3] = p_7[3] + buf[15] + err_3_1;
                            p_7[2] = s_10_1;
                        }
                        if (s0 + 64 < (int)nw[0]) {
                            float s_11 = p_7[0] + buf[16];
                            float bb_4 = s_11 - p_7[0];
                            float err_4 = p_7[0] - (s_11 - bb_4) + (buf[16] - bb_4);
                            p_7[1] = p_7[1] + buf[17] + err_4;
                            p_7[0] = s_11;
                            float s_11_1 = p_7[2] + buf[18];
                            float bb_4_1 = s_11_1 - p_7[2];
                            float err_4_1 = p_7[2] - (s_11_1 - bb_4_1) + (buf[18] - bb_4_1);
                            p_7[3] = p_7[3] + buf[19] + err_4_1;
                            p_7[2] = s_11_1;
                        }
                        if (s0 + 80 < (int)nw[0]) {
                            float s_12 = p_7[0] + buf[20];
                            float bb_5 = s_12 - p_7[0];
                            float err_5 = p_7[0] - (s_12 - bb_5) + (buf[20] - bb_5);
                            p_7[1] = p_7[1] + buf[21] + err_5;
                            p_7[0] = s_12;
                            float s_12_1 = p_7[2] + buf[22];
                            float bb_5_1 = s_12_1 - p_7[2];
                            float err_5_1 = p_7[2] - (s_12_1 - bb_5_1) + (buf[22] - bb_5_1);
                            p_7[3] = p_7[3] + buf[23] + err_5_1;
                            p_7[2] = s_12_1;
                        }
                        if (s0 + 96 < (int)nw[0]) {
                            float s_13 = p_7[0] + buf[24];
                            float bb_6 = s_13 - p_7[0];
                            float err_6 = p_7[0] - (s_13 - bb_6) + (buf[24] - bb_6);
                            p_7[1] = p_7[1] + buf[25] + err_6;
                            p_7[0] = s_13;
                            float s_13_1 = p_7[2] + buf[26];
                            float bb_6_1 = s_13_1 - p_7[2];
                            float err_6_1 = p_7[2] - (s_13_1 - bb_6_1) + (buf[26] - bb_6_1);
                            p_7[3] = p_7[3] + buf[27] + err_6_1;
                            p_7[2] = s_13_1;
                        }
                        if (s0 + 112 < (int)nw[0]) {
                            float s_14 = p_7[0] + buf[28];
                            float bb_7 = s_14 - p_7[0];
                            float err_7 = p_7[0] - (s_14 - bb_7) + (buf[28] - bb_7);
                            p_7[1] = p_7[1] + buf[29] + err_7;
                            p_7[0] = s_14;
                            float s_14_1 = p_7[2] + buf[30];
                            float bb_7_1 = s_14_1 - p_7[2];
                            float err_7_1 = p_7[2] - (s_14_1 - bb_7_1) + (buf[30] - bb_7_1);
                            p_7[3] = p_7[3] + buf[31] + err_7_1;
                            p_7[2] = s_14_1;
                        }
                    }
                    pairs[4 * tid] = p_7[0];
                    pairs[4 * tid + 1] = p_7[1];
                    pairs[4 * tid + 2] = p_7[2];
                    pairs[4 * tid + 3] = p_7[3];
                    asm volatile("barrier.sync 1, 384;" ::: "memory");
                    if (tid < 24) {
                        float s_15 = p_7[0] + pairs[4 * (24 + cl)];
                        float bb_8 = s_15 - p_7[0];
                        float err_8 = p_7[0] - (s_15 - bb_8) + (pairs[4 * (24 + cl)] - bb_8);
                        p_7[1] = p_7[1] + pairs[4 * (24 + cl) + 1] + err_8;
                        p_7[0] = s_15;
                        float s_15_1 = p_7[2] + pairs[4 * (24 + cl) + 2];
                        float bb_8_1 = s_15_1 - p_7[2];
                        float err_8_1 = p_7[2] - (s_15_1 - bb_8_1) + (pairs[4 * (24 + cl) + 2] - bb_8_1);
                        p_7[3] = p_7[3] + pairs[4 * (24 + cl) + 3] + err_8_1;
                        p_7[2] = s_15_1;
                        float s_16 = p_7[0] + pairs[4 * (48 + cl)];
                        float bb_9 = s_16 - p_7[0];
                        float err_9 = p_7[0] - (s_16 - bb_9) + (pairs[4 * (48 + cl)] - bb_9);
                        p_7[1] = p_7[1] + pairs[4 * (48 + cl) + 1] + err_9;
                        p_7[0] = s_16;
                        float s_17 = p_7[2] + pairs[4 * (48 + cl) + 2];
                        float bb_10 = s_17 - p_7[2];
                        float err_10 = p_7[2] - (s_17 - bb_10) + (pairs[4 * (48 + cl) + 2] - bb_10);
                        p_7[3] = p_7[3] + pairs[4 * (48 + cl) + 3] + err_10;
                        p_7[2] = s_17;
                        float s_18 = p_7[0] + pairs[4 * (72 + cl)];
                        float bb_11 = s_18 - p_7[0];
                        float err_11 = p_7[0] - (s_18 - bb_11) + (pairs[4 * (72 + cl)] - bb_11);
                        p_7[1] = p_7[1] + pairs[4 * (72 + cl) + 1] + err_11;
                        p_7[0] = s_18;
                        float s_19 = p_7[2] + pairs[4 * (72 + cl) + 2];
                        float bb_12 = s_19 - p_7[2];
                        float err_12 = p_7[2] - (s_19 - bb_12) + (pairs[4 * (72 + cl) + 2] - bb_12);
                        p_7[3] = p_7[3] + pairs[4 * (72 + cl) + 3] + err_12;
                        p_7[2] = s_19;
                        float s_20 = p_7[0] + pairs[4 * (96 + cl)];
                        float bb_13 = s_20 - p_7[0];
                        float err_13 = p_7[0] - (s_20 - bb_13) + (pairs[4 * (96 + cl)] - bb_13);
                        p_7[1] = p_7[1] + pairs[4 * (96 + cl) + 1] + err_13;
                        p_7[0] = s_20;
                        float s_21 = p_7[2] + pairs[4 * (96 + cl) + 2];
                        float bb_14 = s_21 - p_7[2];
                        float err_14 = p_7[2] - (s_21 - bb_14) + (pairs[4 * (96 + cl) + 2] - bb_14);
                        p_7[3] = p_7[3] + pairs[4 * (96 + cl) + 3] + err_14;
                        p_7[2] = s_21;
                        float s_22 = p_7[0] + pairs[4 * (120 + cl)];
                        float bb_15 = s_22 - p_7[0];
                        float err_15 = p_7[0] - (s_22 - bb_15) + (pairs[4 * (120 + cl)] - bb_15);
                        p_7[1] = p_7[1] + pairs[4 * (120 + cl) + 1] + err_15;
                        p_7[0] = s_22;
                        float s_23 = p_7[2] + pairs[4 * (120 + cl) + 2];
                        float bb_16 = s_23 - p_7[2];
                        float err_16 = p_7[2] - (s_23 - bb_16) + (pairs[4 * (120 + cl) + 2] - bb_16);
                        p_7[3] = p_7[3] + pairs[4 * (120 + cl) + 3] + err_16;
                        p_7[2] = s_23;
                        float s_24 = p_7[0] + pairs[4 * (144 + cl)];
                        float bb_17 = s_24 - p_7[0];
                        float err_17 = p_7[0] - (s_24 - bb_17) + (pairs[4 * (144 + cl)] - bb_17);
                        p_7[1] = p_7[1] + pairs[4 * (144 + cl) + 1] + err_17;
                        p_7[0] = s_24;
                        float s_25 = p_7[2] + pairs[4 * (144 + cl) + 2];
                        float bb_18 = s_25 - p_7[2];
                        float err_18 = p_7[2] - (s_25 - bb_18) + (pairs[4 * (144 + cl) + 2] - bb_18);
                        p_7[3] = p_7[3] + pairs[4 * (144 + cl) + 3] + err_18;
                        p_7[2] = s_25;
                        float s_26 = p_7[0] + pairs[4 * (168 + cl)];
                        float bb_19 = s_26 - p_7[0];
                        float err_19 = p_7[0] - (s_26 - bb_19) + (pairs[4 * (168 + cl)] - bb_19);
                        p_7[1] = p_7[1] + pairs[4 * (168 + cl) + 1] + err_19;
                        p_7[0] = s_26;
                        float s_27 = p_7[2] + pairs[4 * (168 + cl) + 2];
                        float bb_20 = s_27 - p_7[2];
                        float err_20 = p_7[2] - (s_27 - bb_20) + (pairs[4 * (168 + cl) + 2] - bb_20);
                        p_7[3] = p_7[3] + pairs[4 * (168 + cl) + 3] + err_20;
                        p_7[2] = s_27;
                        float s_28 = p_7[0] + pairs[4 * (192 + cl)];
                        float bb_21 = s_28 - p_7[0];
                        float err_21 = p_7[0] - (s_28 - bb_21) + (pairs[4 * (192 + cl)] - bb_21);
                        p_7[1] = p_7[1] + pairs[4 * (192 + cl) + 1] + err_21;
                        p_7[0] = s_28;
                        float s_29 = p_7[2] + pairs[4 * (192 + cl) + 2];
                        float bb_22 = s_29 - p_7[2];
                        float err_22 = p_7[2] - (s_29 - bb_22) + (pairs[4 * (192 + cl) + 2] - bb_22);
                        p_7[3] = p_7[3] + pairs[4 * (192 + cl) + 3] + err_22;
                        p_7[2] = s_29;
                        float s_30 = p_7[0] + pairs[4 * (216 + cl)];
                        float bb_23 = s_30 - p_7[0];
                        float err_23 = p_7[0] - (s_30 - bb_23) + (pairs[4 * (216 + cl)] - bb_23);
                        p_7[1] = p_7[1] + pairs[4 * (216 + cl) + 1] + err_23;
                        p_7[0] = s_30;
                        float s_31 = p_7[2] + pairs[4 * (216 + cl) + 2];
                        float bb_24 = s_31 - p_7[2];
                        float err_24 = p_7[2] - (s_31 - bb_24) + (pairs[4 * (216 + cl) + 2] - bb_24);
                        p_7[3] = p_7[3] + pairs[4 * (216 + cl) + 3] + err_24;
                        p_7[2] = s_31;
                        float s_32 = p_7[0] + pairs[4 * (240 + cl)];
                        float bb_25 = s_32 - p_7[0];
                        float err_25 = p_7[0] - (s_32 - bb_25) + (pairs[4 * (240 + cl)] - bb_25);
                        p_7[1] = p_7[1] + pairs[4 * (240 + cl) + 1] + err_25;
                        p_7[0] = s_32;
                        float s_33 = p_7[2] + pairs[4 * (240 + cl) + 2];
                        float bb_26 = s_33 - p_7[2];
                        float err_26 = p_7[2] - (s_33 - bb_26) + (pairs[4 * (240 + cl) + 2] - bb_26);
                        p_7[3] = p_7[3] + pairs[4 * (240 + cl) + 3] + err_26;
                        p_7[2] = s_33;
                        float s_34 = p_7[0] + pairs[4 * (264 + cl)];
                        float bb_27 = s_34 - p_7[0];
                        float err_27 = p_7[0] - (s_34 - bb_27) + (pairs[4 * (264 + cl)] - bb_27);
                        p_7[1] = p_7[1] + pairs[4 * (264 + cl) + 1] + err_27;
                        p_7[0] = s_34;
                        float s_35 = p_7[2] + pairs[4 * (264 + cl) + 2];
                        float bb_28 = s_35 - p_7[2];
                        float err_28 = p_7[2] - (s_35 - bb_28) + (pairs[4 * (264 + cl) + 2] - bb_28);
                        p_7[3] = p_7[3] + pairs[4 * (264 + cl) + 3] + err_28;
                        p_7[2] = s_35;
                        float s_36 = p_7[0] + pairs[4 * (288 + cl)];
                        float bb_29 = s_36 - p_7[0];
                        float err_29 = p_7[0] - (s_36 - bb_29) + (pairs[4 * (288 + cl)] - bb_29);
                        p_7[1] = p_7[1] + pairs[4 * (288 + cl) + 1] + err_29;
                        p_7[0] = s_36;
                        float s_37 = p_7[2] + pairs[4 * (288 + cl) + 2];
                        float bb_30 = s_37 - p_7[2];
                        float err_30 = p_7[2] - (s_37 - bb_30) + (pairs[4 * (288 + cl) + 2] - bb_30);
                        p_7[3] = p_7[3] + pairs[4 * (288 + cl) + 3] + err_30;
                        p_7[2] = s_37;
                        float s_38 = p_7[0] + pairs[4 * (312 + cl)];
                        float bb_31 = s_38 - p_7[0];
                        float err_31 = p_7[0] - (s_38 - bb_31) + (pairs[4 * (312 + cl)] - bb_31);
                        p_7[1] = p_7[1] + pairs[4 * (312 + cl) + 1] + err_31;
                        p_7[0] = s_38;
                        float s_39 = p_7[2] + pairs[4 * (312 + cl) + 2];
                        float bb_32 = s_39 - p_7[2];
                        float err_32 = p_7[2] - (s_39 - bb_32) + (pairs[4 * (312 + cl) + 2] - bb_32);
                        p_7[3] = p_7[3] + pairs[4 * (312 + cl) + 3] + err_32;
                        p_7[2] = s_39;
                        float s_40 = p_7[0] + pairs[4 * (336 + cl)];
                        float bb_33 = s_40 - p_7[0];
                        float err_33 = p_7[0] - (s_40 - bb_33) + (pairs[4 * (336 + cl)] - bb_33);
                        p_7[1] = p_7[1] + pairs[4 * (336 + cl) + 1] + err_33;
                        p_7[0] = s_40;
                        float s_41 = p_7[2] + pairs[4 * (336 + cl) + 2];
                        float bb_34 = s_41 - p_7[2];
                        float err_34 = p_7[2] - (s_41 - bb_34) + (pairs[4 * (336 + cl) + 2] - bb_34);
                        p_7[3] = p_7[3] + pairs[4 * (336 + cl) + 3] + err_34;
                        p_7[2] = s_41;
                        float s_42 = p_7[0] + pairs[4 * (360 + cl)];
                        float bb_35 = s_42 - p_7[0];
                        float err_35 = p_7[0] - (s_42 - bb_35) + (pairs[4 * (360 + cl)] - bb_35);
                        p_7[1] = p_7[1] + pairs[4 * (360 + cl) + 1] + err_35;
                        p_7[0] = s_42;
                        float s_43 = p_7[2] + pairs[4 * (360 + cl) + 2];
                        float bb_36 = s_43 - p_7[2];
                        float err_36 = p_7[2] - (s_43 - bb_36) + (pairs[4 * (360 + cl) + 2] - bb_36);
                        p_7[3] = p_7[3] + pairs[4 * (360 + cl) + 3] + err_36;
                        p_7[2] = s_43;
                        *(reinterpret_cast<float*>(dw) + ((at0 / 384 / 8 * 384 + at0 % 384) * 8 + at0 / 384 % 8)) = p_7[0] + p_7[1];
                        *(reinterpret_cast<float*>(dw) + (((at0 + 1) / 384 / 8 * 384 + (at0 + 1) % 384) * 8 + (at0 + 1) / 384 % 8)) = p_7[2] + p_7[3];
                    }
                    asm volatile("barrier.sync 1, 384;" ::: "memory");
                }
                if (tid == 0) {
                    int add2 = 1;
                    if (hs_c == 0) {
                        add2 = 129 - helpers;
                    }
                    unsigned int old2;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(old2) : "l"(&counters[2]), "r"(static_cast<uint32_t>(add2)) : "memory");
                    if (old2 + (unsigned int)add2 == 128) {
                        *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters) + (2)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters) + (3)) = 0;
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
                    int sl_1 = rows_per_chunk;
                    if (sl_1 > stages) {
                        sl_1 = stages;
                    }
                    if (sl_1 < 1) {
                        sl_1 = 1;
                    }
                    int share = (stages + sl_1 - 1) / sl_1;
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
                                if (rem >= sl_1 * size) {
                                    if (p_c < first + sl_1) {
                                        span = size;
                                        begin_it = base + (p_c - first) * size;
                                        halving = 0;
                                    } else {
                                        first = first + sl_1;
                                        base = base + sl_1 * size;
                                        rem = rem - sl_1 * size;
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
                        int quot = rem / sl_1;
                        int extra = rem - quot * sl_1;
                        if (p_c < first + extra) {
                            span = quot + 1;
                            begin_it = base + (p_c - first) * (quot + 1);
                        } else if (p_c < first + sl_1) {
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
                    int slot_1 = pk % 4;
                    if (p_row >= p_end) {
                        if (p_c < n_chunks) {
                            p_c = (int)p_nxt;
                            if (p_c < n_chunks) {
                                int stages_1 = T;
                                int sl_2 = rows_per_chunk;
                                if (sl_2 > stages_1) {
                                    sl_2 = stages_1;
                                }
                                if (sl_2 < 1) {
                                    sl_2 = 1;
                                }
                                int share_1 = (stages_1 + sl_2 - 1) / sl_2;
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
                                            if (rem_1 >= sl_2 * size_1) {
                                                if (p_c < first_1 + sl_2) {
                                                    span_1 = size_1;
                                                    begin_it_1 = base_1 + (p_c - first_1) * size_1;
                                                    halving_1 = 0;
                                                } else {
                                                    first_1 = first_1 + sl_2;
                                                    base_1 = base_1 + sl_2 * size_1;
                                                    rem_1 = rem_1 - sl_2 * size_1;
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
                                    int quot_1 = rem_1 / sl_2;
                                    int extra_1 = rem_1 - quot_1 * sl_2;
                                    if (p_c < first_1 + extra_1) {
                                        span_1 = quot_1 + 1;
                                        begin_it_1 = base_1 + (p_c - first_1) * (quot_1 + 1);
                                    } else if (p_c < first_1 + sl_2) {
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
                        meta[2 * slot_1] = p_row;
                        meta[2 * slot_1 + 1] = nval_1;
                        mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)pk % 4)) * 8, 24576);
                        int sbase = p_ring + slot_1 * 24576;
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
                    } else {
                        meta[2 * slot_1] = p_c;
                        meta[2 * slot_1 + 1] = -1 - p_first;
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
