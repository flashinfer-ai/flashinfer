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
#define SMEM_DOT_SLOTS_OFF 0
#define SMEM_DOT_SLOTS_STAGE_BYTES 32
#define SMEM_DOT_SLOTS_STRIDE 32
#define SMEM_FLAGS_OFF 128
#define SMEM_FLAGS_STAGE_BYTES 32
#define SMEM_FLAGS_STRIDE 32
#define SMEM_RED_FOLD_OFF 256
#define SMEM_RED_FOLD_STAGE_BYTES 2048
#define SMEM_RED_FOLD_STRIDE 2048
#define SMEM_ROW_FOLD_OFF 2304
#define SMEM_ROW_FOLD_STAGE_BYTES 4096
#define SMEM_ROW_FOLD_STRIDE 4096
#define SMEM_COMP_SMEM_OFF 6400
#define SMEM_COMP_SMEM_STAGE_BYTES 128
#define SMEM_COMP_SMEM_STRIDE 128
#define SMEM_W_SM_OFF 6528
#define SMEM_W_SM_STAGE_BYTES 16
#define SMEM_W_SM_STRIDE 16
#define SMEM_PART_SM_OFF 6656
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_RING_OFF 6784
#define SMEM_RING_STAGE_BYTES 36864
#define SMEM_RING_STRIDE 36864
#define SMEM_TOTAL 43648
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 5

#include <math_constants.h>

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

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_c18297d75c0d424978fd(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* dot_slots = reinterpret_cast<float*>(smem_raw + 0);
    const int dot_slots_addr = smem + 0;
    int* flags = reinterpret_cast<int*>(smem_raw + 128);
    const int flags_addr = smem + 128;
    float* red_fold = reinterpret_cast<float*>(smem_raw + 256);
    const int red_fold_addr = smem + 256;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 2304);
    const int row_fold_addr = smem + 2304;
    float* comp_smem = reinterpret_cast<float*>(smem_raw + 6400);
    const int comp_smem_addr = smem + 6400;
    unsigned int* w_sm = reinterpret_cast<unsigned int*>(smem_raw + 6528);
    const int w_sm_addr = smem + 6528;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 6656);
    const int part_sm_addr = smem + 6656;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 6784);
    const int ring_addr = smem + 6784;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = 0;
    int rg = warp;
    int tir = tid - rg * 32;
    int row_begin = 0;
    int row_end = 0;
    int n_rows = 0;
    int n_iters = 0;
    int last_row = 0;
    float wc[8];
    unsigned int ww[4];
    #pragma unroll
    for (int v = 0; v < 1; v++) {
        int wcolp = (v * 32 + tir) * 8;
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcolp);
            uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_1[_blk] = _vptr_1[_blk];
            }
        }
    }
    float acc[8];
    float comp[8];
    unsigned int compw[4];
    if (tid == 0) {
        flags[1] = (int)nxt;
    }
    __syncthreads();
    c = flags[1];
    #pragma unroll 1
    for (int claim_it = 0; claim_it < n_chunks; claim_it++) {
        if (c >= n_chunks) {
            break;
        }
        if (tid == 0) {
            unsigned int _atomic_fetch_add_2 = atomicAdd(&counters[0], 1);
            nxt = _atomic_fetch_add_2;
        }
        int stages = (T + 3) / 4;
        int sl = rows_per_chunk;
        if (sl > stages) {
            sl = stages;
        }
        if (sl < 1) {
            sl = 1;
        }
        int share = (stages + sl - 1) / sl;
        int tail = (share + 3) / 4;
        if (tail < 1) {
            tail = 1;
        }
        int rem = stages;
        int size = (share + 1) / 2;
        int first = 0;
        int base = 0;
        int rows = 0;
        int halving = 1;
        int begin_it = 0;
        #pragma unroll 1
        for (int k = 0; k < 5; k++) {
            if (halving != 0) {
                if (size > tail) {
                    if (rem >= sl * size) {
                        if (c < first + sl) {
                            rows = size;
                            begin_it = base + (c - first);
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
        if (rows == 0) {
            int quot = rem / sl;
            int extra = rem - quot * sl;
            if (c < first + extra) {
                rows = quot + 1;
                begin_it = base + (c - first);
            } else if (c < first + sl) {
                rows = quot;
                begin_it = base + (c - first);
            } else {
                begin_it = stages;
            }
        }
        row_begin = begin_it * 4;
        row_end = row_begin + rows * 4;
        if (row_begin > T) {
            row_begin = T;
        }
        if (row_end > T) {
            row_end = T;
        }
        n_rows = row_end - row_begin;
        n_iters = (n_rows + 3) / 4;
        last_row = row_end - 1;
        int rstep = sl * 4;
        row_end = T;
        n_iters = rows;
        last_row = T - 1;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j] = 0.0f;
            comp[j] = 0.0f;
        }
        unsigned int gwA[4];
        unsigned int xwA[4];
        unsigned int ghwA[4];
        float rrA[1];
        unsigned int gwB[4];
        unsigned int xwB[4];
        unsigned int ghwB[4];
        float rrB[1];
        if (n_iters > 0) {
            int row0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
            unsigned int sg = (unsigned int)ring_addr + (unsigned int)(rg * 9216) + (unsigned int)(tir * 16);
            unsigned int sr = (unsigned int)ring_addr + (unsigned int)(rg * 9216 + 1024) + (unsigned int)(tir * 4);
            #pragma unroll
            for (int p = 0; p < 3; p++) {
                int rowq_ld = ((row_begin + p * rstep + rg) < (last_row) ? (row_begin + p * rstep + rg) : (last_row));
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((p < n_iters) ? 1 : 0), "r"(sg + (unsigned int)(p * 1152)), "l"(g + ((unsigned long long)rowq_ld * (unsigned long long)g_stride + (unsigned long long)(tir * 8))));
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((p < n_iters) ? 1 : 0), "r"(sg + (unsigned int)(p * 1152 + 512)), "l"(x + ((unsigned long long)rowq_ld * (unsigned long long)x_stride + (unsigned long long)(tir * 8))));
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                    "}"
                    :: "r"((p < n_iters) ? 1 : 0), "r"(sr + (unsigned int)(p * 1152)), "l"(r + rowq_ld));
                asm volatile("cp.async.commit_group;");
            }
            #pragma unroll 1
            for (int it = 0; it < n_iters; it++) {
                int nx = it + 3;
                int rowx_ld = ((row_begin + nx * rstep + rg) < (last_row) ? (row_begin + nx * rstep + rg) : (last_row));
                unsigned int slx = (unsigned int)(nx & 7) * 1152;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((nx < n_iters) ? 1 : 0), "r"(sg + slx), "l"(g + ((unsigned long long)rowx_ld * (unsigned long long)g_stride + (unsigned long long)(tir * 8))));
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((nx < n_iters) ? 1 : 0), "r"(sg + slx + 512), "l"(x + ((unsigned long long)rowx_ld * (unsigned long long)x_stride + (unsigned long long)(tir * 8))));
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                    "}"
                    :: "r"((nx < n_iters) ? 1 : 0), "r"(sr + slx), "l"(r + rowx_ld));
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 3;");
                unsigned int slc = (unsigned int)(it & 7) * 1152;
                __int128_t _smem_raw_3;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_3) : "r"(sg + slc));
                gwA[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[0];
                gwA[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[1];
                gwA[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[2];
                gwA[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_3)[3];
                __int128_t _smem_raw_4;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_4) : "r"(sg + slc + 512));
                xwA[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[0];
                xwA[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[1];
                xwA[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[2];
                xwA[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_4)[3];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&rrA[0])) : "r"(sr + slc));
                int row6a = row_begin + it * rstep + rg;
                float dq = 0.0f;
                float gwr[8];
                #pragma unroll
                for (int v_1 = 0; v_1 < 1; v_1++) {
                    unsigned int wvd[4];
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        unsigned int gwd = gwA[v_1 * 4 + e];
                        unsigned int xwd = xwA[v_1 * 4 + e];
                        float g0 = __uint_as_float(gwd << 16);
                        float g1 = __uint_as_float(gwd & 4294901760u);
                        float x0 = __uint_as_float(xwd << 16);
                        float x1 = __uint_as_float(xwd & 4294901760u);
                        float2 wq = make_float2(__uint_as_float(ww[v_1 * 4 + e] << 16), __uint_as_float(ww[v_1 * 4 + e] & 4294901760u));
                        float2 _f32x2_mul_a_5 = make_float2(g0, g1);
                        float2 gwq;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gwq) : "l"(*(const unsigned long long*)&_f32x2_mul_a_5), "l"(*(const unsigned long long*)&wq));
                        gwr[v_1 * 8 + 2 * e] = gwq.x;
                        gwr[v_1 * 8 + 2 * e + 1] = gwq.y;
                        dq = __fmaf_rn(gwq.x, x0, dq);
                        dq = __fmaf_rn(gwq.y, x1, dq);
                    }
                }
                float dots[1];
                float _warp_reduce_6 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_6 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_6, offset);
                dots[0] = _warp_reduce_6;
                float rq = rrA[0];
                float cq = __fdiv_rn(dots[0], 256.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row6a < row_end) {
                    unsigned long long dx_base = (unsigned long long)row6a * 256;
                    float2 rr = make_float2(rq, rq);
                    float2 rrn = make_float2(-rq, -rq);
                    float2 nn = make_float2(nkx, nkx);
                    #pragma unroll
                    for (int v_2 = 0; v_2 < 1; v_2++) {
                        float out[8];
                        unsigned int wv[4];
                        #pragma unroll
                        for (int e_1 = 0; e_1 < 4; e_1++) {
                            unsigned int gwd_1 = gwA[v_2 * 4 + e_1];
                            unsigned int xwd_1 = xwA[v_2 * 4 + e_1];
                            float g0_1 = __uint_as_float(gwd_1 << 16);
                            float g1_1 = __uint_as_float(gwd_1 & 4294901760u);
                            float x0_1 = __uint_as_float(xwd_1 << 16);
                            float x1_1 = __uint_as_float(xwd_1 & 4294901760u);
                            float2 _f32x2_mul_a_7 = make_float2(g0_1, g1_1);
                            float2 _f32x2_mul_b_8 = make_float2(x0_1, x1_1);
                            float2 gx;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                            float2 pn;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&pn) : "l"(*(const unsigned long long*)&gx), "l"(*(const unsigned long long*)&rrn));
                            float2 pe = fma_f32x2_rn_noftz(gx, rr, pn);
                            float2 a2 = make_float2(acc[v_2 * 8 + 2 * e_1], acc[v_2 * 8 + 2 * e_1 + 1]);
                            float2 s2 = sub_f32x2_noftz(a2, pn);
                            float2 z2 = sub_f32x2_noftz(s2, a2);
                            float2 t2 = sub_f32x2_noftz(pe, add_f32x2_noftz(pn, z2));
                            float2 c2 = add_f32x2_noftz(make_float2(comp[v_2 * 8 + 2 * e_1], comp[v_2 * 8 + 2 * e_1 + 1]), t2);
                            acc[v_2 * 8 + 2 * e_1] = s2.x;
                            acc[v_2 * 8 + 2 * e_1 + 1] = s2.y;
                            comp[v_2 * 8 + 2 * e_1] = c2.x;
                            comp[v_2 * 8 + 2 * e_1 + 1] = c2.y;
                            float2 _f32x2_mul_b_9 = make_float2(gwr[v_2 * 8 + 2 * e_1], gwr[v_2 * 8 + 2 * e_1 + 1]);
                            float2 q2;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&q2) : "l"(*(const unsigned long long*)&rr), "l"(*(const unsigned long long*)&_f32x2_mul_b_9));
                            float2 o2 = fma_f32x2_rn_noftz(nn, make_float2(x0_1, x1_1), q2);
                            out[2 * e_1] = o2.x;
                            out[2 * e_1 + 1] = o2.y;
                        }
                        int col = (v_2 * 32 + tir) * 8;
                        unsigned int packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                            packed[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                            packed[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                            packed[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;");
        }
        unsigned long long pbase = (unsigned long long)c * 256;
        #pragma unroll
        for (int v_3 = 0; v_3 < 1; v_3++) {
            int fcol = (v_3 * 32 + tir) * 8;
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2++) {
                row_fold[rg * 256 + fcol + e_2] = acc[v_3 * 8 + e_2];
            }
        }
        __syncthreads();
        float fold_acc[2];
        #pragma unroll
        for (int e_3 = 0; e_3 < 2; e_3++) {
            fold_acc[e_3] = 0.0f;
        }
        float fold_c[2];
        #pragma unroll
        for (int e_4 = 0; e_4 < 2; e_4++) {
            fold_c[e_4] = 0.0f;
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            #pragma unroll
            for (int e_5 = 0; e_5 < 2; e_5++) {
                float d2 = row_fold[k_1 * 256 + tid * 2 + e_5];
                float fs = fold_acc[e_5] + d2;
                float fb = fs - fold_acc[e_5];
                fold_c[e_5] = fold_c[e_5] + (fold_acc[e_5] - (fs - fb) + (d2 - fb));
                fold_acc[e_5] = fs;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int e5 = 0; e5 < 8; e5++) {
            row_fold[rg * 256 + tir * 8 + e5] = comp[e5];
        }
        __syncthreads();
        #pragma unroll
        for (int k5 = 0; k5 < 4; k5++) {
            #pragma unroll
            for (int e_6 = 0; e_6 < 2; e_6++) {
                float d5 = row_fold[k5 * 256 + tid * 2 + e_6];
                fold_c[e_6] = fold_c[e_6] + d5;
            }
        }
        int pcol = tid * 2;
        {
            float2 _v2 = make_float2(fold_acc[0 + 0], fold_acc[0 + 1]);
            *reinterpret_cast<float2*>(partial + pbase + (unsigned long long)pcol) = _v2;
        }
        unsigned long long lbase = (unsigned long long)(n_chunks + c) * 256;
        {
            float2 _v2 = make_float2(fold_c[0 + 0], fold_c[0 + 1]);
            *reinterpret_cast<float2*>(partial + lbase + (unsigned long long)pcol) = _v2;
        }
        if (tid == 0) {
            flags[claim_it & 1] = (int)nxt;
        }
        __syncthreads();
        c = flags[claim_it & 1];
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid == 0) {
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
}

} // extern "C"
