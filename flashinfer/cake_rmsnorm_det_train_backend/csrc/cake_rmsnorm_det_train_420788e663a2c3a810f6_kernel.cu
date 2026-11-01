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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_det_train_420788e663a2c3a810f6(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, float* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    #pragma unroll
    for (int v = 0; v < 1; v++) {
        int wcol = (v * 32 + tir) * 8;
        {
            unsigned _ldv8_1_0;
            unsigned _ldv8_1_1;
            unsigned _ldv8_1_2;
            unsigned _ldv8_1_3;
            unsigned _ldv8_1_4;
            unsigned _ldv8_1_5;
            unsigned _ldv8_1_6;
            unsigned _ldv8_1_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(w + ((unsigned long long)wcol))) : "memory");
            wc[v * 8 + 0] = __uint_as_float(_ldv8_1_0);
            wc[v * 8 + 1] = __uint_as_float(_ldv8_1_1);
            wc[v * 8 + 2] = __uint_as_float(_ldv8_1_2);
            wc[v * 8 + 3] = __uint_as_float(_ldv8_1_3);
            wc[v * 8 + 4] = __uint_as_float(_ldv8_1_4);
            wc[v * 8 + 5] = __uint_as_float(_ldv8_1_5);
            wc[v * 8 + 6] = __uint_as_float(_ldv8_1_6);
            wc[v * 8 + 7] = __uint_as_float(_ldv8_1_7);
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
                        dq = __fmaf_rn(g0 * wc[v_1 * 8 + 2 * e], x0, dq);
                        dq = __fmaf_rn(g1 * wc[v_1 * 8 + 2 * e + 1], x1, dq);
                    }
                }
                float dots[1];
                float _warp_reduce_5 = dq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_5 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_5, offset);
                dots[0] = _warp_reduce_5;
                float rq = rrA[0];
                float cq = __fdiv_rn(dots[0], 256.0f);
                float kx = rq * rq * rq * cq;
                float nkx = -kx;
                if (row6a < row_end) {
                    unsigned long long dx_base = (unsigned long long)row6a * 256;
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
                            acc[v_2 * 8 + 2 * e_1] = __fmaf_rn(g0_1 * x0_1, rq, acc[v_2 * 8 + 2 * e_1]);
                            acc[v_2 * 8 + 2 * e_1 + 1] = __fmaf_rn(g1_1 * x1_1, rq, acc[v_2 * 8 + 2 * e_1 + 1]);
                            out[2 * e_1] = __fmaf_rn(nkx, x0_1, rq * (g0_1 * wc[v_2 * 8 + 2 * e_1]));
                            out[2 * e_1 + 1] = __fmaf_rn(nkx, x1_1, rq * (g1_1 * wc[v_2 * 8 + 2 * e_1 + 1]));
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
                if ((it & 7) == 7) {
                    #pragma unroll
                    for (int jc = 0; jc < 8; jc++) {
                        comp[jc] = comp[jc] + acc[jc];
                        acc[jc] = 0.0f;
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;");
        }
        #pragma unroll
        for (int j3 = 0; j3 < 8; j3++) {
            acc[j3] = comp[j3] + acc[j3];
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
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 2; e_4++) {
                float d2 = row_fold[k_1 * 256 + tid * 2 + e_4];
                fold_acc[e_4] = fold_acc[e_4] + d2;
            }
        }
        int pcol = tid * 2;
        {
            float2 _v2 = make_float2(fold_acc[0 + 0], fold_acc[0 + 1]);
            *reinterpret_cast<float2*>(partial + pbase + (unsigned long long)pcol) = _v2;
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
