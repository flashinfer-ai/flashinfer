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
#define NUM_MAIN_STAGES 1
#define SMEM_STAGE_BUF_OFF 1024
#define SMEM_STAGE_BUF_STAGE_BYTES 98304
#define SMEM_STAGE_BUF_STRIDE 98304
#define SMEM_ROW_FOLD_OFF 1024
#define SMEM_ROW_FOLD_STAGE_BYTES 128
#define SMEM_ROW_FOLD_STRIDE 128
#define SMEM_RED_FOLD_OFF 1152
#define SMEM_RED_FOLD_STAGE_BYTES 6144
#define SMEM_RED_FOLD_STRIDE 6144
#define SMEM_DOT_SLOTS_OFF 99328
#define SMEM_DOT_SLOTS_STAGE_BYTES 96
#define SMEM_DOT_SLOTS_STRIDE 96
#define SMEM_PART_SM_OFF 99456
#define SMEM_PART_SM_STAGE_BYTES 24576
#define SMEM_PART_SM_STRIDE 24576
#define SMEM_GRP_FOLD_OFF 7296
#define SMEM_GRP_FOLD_STAGE_BYTES 1024
#define SMEM_GRP_FOLD_STRIDE 1024
#define SMEM_TOTAL 124032
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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_89789ed929368872c1c1(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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
    #define bars_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int stage_buf_addr = smem + 1024;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_fold_addr = smem + 1024;
    float* red_fold = reinterpret_cast<float*>(smem_raw + 1152);
    const int red_fold_addr = smem + 1152;
    float* dot_slots = reinterpret_cast<float*>(smem_raw + 99328);
    const int dot_slots_addr = smem + 99328;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 99456);
    const int part_sm_addr = smem + 99456;
    float* grp_fold = reinterpret_cast<float*>(smem_raw + 7296);
    const int grp_fold_addr = smem + 7296;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // bars: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = bid;
    int rg = tid / 384;
    int tir = tid - rg * 384;
    int row_begin = c * rows_per_chunk;
    int row_end = row_begin + rows_per_chunk;
    if (row_end > T) {
        row_end = T;
    }
    if (row_begin > T) {
        row_begin = T;
    }
    int n_rows = row_end - row_begin;
    int n_steps = n_rows;
    int last_row = row_end - 1;
    int ring = (int)stage_buf_addr;
    int my_g = ring + rg * 12288 + tir * 16;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        if (n_steps > s) {
            int first = row_begin + s;
            int rows_k = ((row_end - first) < (1) ? (row_end - first) : (1));
            int sbase = ring + s * 24576;
            if (tid == 0) {
                mbarrier_arrive_expect_tx(bars_addr + (s) * 8, rows_k * 24576);
            }
            if (tid == 0) {
                #pragma unroll
                for (int i = 0; i < 1; i++) {
                    if (rows_k > i) {
                        int row_i = first + i;
                        long long g_src = (long long)row_i * g_stride;
                        long long x_src = (long long)row_i * x_stride;
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + i * 12288), "l"(reinterpret_cast<const uint8_t*>(g + g_src)), "r"((unsigned)(12288)), "r"(bars_addr + (s) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase + (1 + i) * 12288), "l"(reinterpret_cast<const uint8_t*>(x + x_src)), "r"((unsigned)(12288)), "r"(bars_addr + (s) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
        }
    }
    unsigned int flip_old[1];
    unsigned int ww[8];
    #pragma unroll
    for (int v = 0; v < 2; v++) {
        int wcol = (v * 384 + tir) * 8;
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcol);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
    }
    float wf[16];
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        unsigned int wwd = ww[p];
        wf[2 * p] = __uint_as_float(wwd << 16);
        wf[2 * p + 1] = __uint_as_float(wwd & 4294901760u);
    }
    float acc[16];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        acc[j] = 0.0f;
    }
    #pragma unroll
    for (int jp0 = 0; jp0 < 16; jp0++) {
        part_sm[jp0 * 384 + tid] = 0.0f;
    }
    float r_cur = 0.0f;
    if (n_steps > 0) {
        int row0 = ((row_begin + rg) < (last_row) ? (row_begin + rg) : (last_row));
        r_cur = r[row0];
    }
    unsigned int gw[8];
    unsigned int xw[8];
    float out[16];
    #pragma unroll 1
    for (int k = 0; k < n_steps; k++) {
        int slot = k % 4;
        int parity = k / 4 & 1;
        int row = row_begin + k + rg;
        int row_n = ((row + 1) < (last_row) ? (row + 1) : (last_row));
        float r_nxt = r[row_n];
        mbarrier_wait(bars_addr + (slot) * 8, parity);
        int sg = my_g + slot * 24576;
        #pragma unroll
        for (int v_1 = 0; v_1 < 2; v_1++) {
            __int128_t _smem_raw_1;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_1) : "r"(sg + v_1 * 6144));
            gw[v_1 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[0];
            gw[(v_1 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[1];
            gw[(v_1 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[2];
            gw[(v_1 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[3];
            __int128_t _smem_raw_2;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_2) : "r"(sg + (12288 + v_1 * 6144)));
            xw[v_1 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[0];
            xw[(v_1 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[1];
            xw[(v_1 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[2];
            xw[(v_1 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_2)[3];
        }
        float dq = 0.0f;
        #pragma unroll
        for (int v_2 = 0; v_2 < 2; v_2++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                unsigned int gwd = gw[v_2 * 4 + e];
                unsigned int xwd = xw[v_2 * 4 + e];
                float g0 = __uint_as_float(gwd << 16);
                float g1 = __uint_as_float(gwd & 4294901760u);
                float x0 = __uint_as_float(xwd << 16);
                float x1 = __uint_as_float(xwd & 4294901760u);
                dq = __fmaf_rn(g0 * wf[v_2 * 8 + 2 * e], x0, dq);
                dq = __fmaf_rn(g1 * wf[v_2 * 8 + 2 * e + 1], x1, dq);
            }
        }
        float dot = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        int par = (k & 1) * 12;
        if (lane == 0) {
            dot_slots[par + warp] = dot;
        }
        __syncthreads();
        int kn = k + 4;
        if (kn < n_steps) {
            int first_1 = row_begin + kn;
            int rows_k_1 = ((row_end - first_1) < (1) ? (row_end - first_1) : (1));
            int sbase_1 = ring + slot * 24576;
            if (tid == 0) {
                mbarrier_arrive_expect_tx(bars_addr + (slot) * 8, rows_k_1 * 24576);
            }
            if (tid == 0) {
                #pragma unroll
                for (int i_1 = 0; i_1 < 1; i_1++) {
                    if (rows_k_1 > i_1) {
                        int row_i_1 = first_1 + i_1;
                        long long g_src_1 = (long long)row_i_1 * g_stride;
                        long long x_src_1 = (long long)row_i_1 * x_stride;
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase_1 + i_1 * 12288), "l"(reinterpret_cast<const uint8_t*>(g + g_src_1)), "r"((unsigned)(12288)), "r"(bars_addr + (slot) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase_1 + (1 + i_1) * 12288), "l"(reinterpret_cast<const uint8_t*>(x + x_src_1)), "r"((unsigned)(12288)), "r"(bars_addr + (slot) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
            }
        }
        float tot = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < 12; kk++) {
            float d = dot_slots[par + rg * 12 + kk];
            tot += d;
        }
        float rq = r_cur;
        float cq = __fdiv_rn(tot, 6144.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row < row_end) {
            unsigned long long dx_base = (unsigned long long)row * 6144;
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    unsigned int gwd2 = gw[v_3 * 4 + e_1];
                    unsigned int xwd2 = xw[v_3 * 4 + e_1];
                    float g2 = __uint_as_float(gwd2 << 16);
                    float g3 = __uint_as_float(gwd2 & 4294901760u);
                    float x2 = __uint_as_float(xwd2 << 16);
                    float x3 = __uint_as_float(xwd2 & 4294901760u);
                    acc[v_3 * 8 + 2 * e_1] = __fmaf_rn(g2 * x2, rq, acc[v_3 * 8 + 2 * e_1]);
                    acc[v_3 * 8 + 2 * e_1 + 1] = __fmaf_rn(g3 * x3, rq, acc[v_3 * 8 + 2 * e_1 + 1]);
                    out[v_3 * 8 + 2 * e_1] = __fmaf_rn(nkx, x2, rq * (g2 * wf[v_3 * 8 + 2 * e_1]));
                    out[v_3 * 8 + 2 * e_1 + 1] = __fmaf_rn(nkx, x3, rq * (g3 * wf[v_3 * 8 + 2 * e_1 + 1]));
                }
                int dcol = (v_3 * 384 + tir) * 8;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(out[v_3 * 8 + 0], out[v_3 * 8 + 1]);
                    _pk[1] = __floats2bfloat162_rn(out[v_3 * 8 + 2], out[v_3 * 8 + 3]);
                    _pk[2] = __floats2bfloat162_rn(out[v_3 * 8 + 4], out[v_3 * 8 + 5]);
                    _pk[3] = __floats2bfloat162_rn(out[v_3 * 8 + 6], out[v_3 * 8 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(dx))[dx_base + (unsigned long long)dcol + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
        if ((k + 1 & 7) == 0) {
            #pragma unroll
            for (int jf = 0; jf < 16; jf++) {
                float pf = part_sm[jf * 384 + tid];
                part_sm[jf * 384 + tid] = pf + acc[jf];
                acc[jf] = 0.0f;
            }
        }
        r_cur = r_nxt;
    }
    #pragma unroll
    for (int jp = 0; jp < 16; jp++) {
        float pfin = part_sm[jp * 384 + tid];
        acc[jp] = pfin + acc[jp];
    }
    unsigned long long pbase = (unsigned long long)c * 6144;
    #pragma unroll
    for (int v_4 = 0; v_4 < 2; v_4++) {
        int pcolv = (v_4 * 384 + tir) * 8;
        {
            unsigned _stv8_3_0 = __float_as_uint(acc[v_4 * 8 + 0]);
            unsigned _stv8_3_1 = __float_as_uint(acc[v_4 * 8 + 1]);
            unsigned _stv8_3_2 = __float_as_uint(acc[v_4 * 8 + 2]);
            unsigned _stv8_3_3 = __float_as_uint(acc[v_4 * 8 + 3]);
            unsigned _stv8_3_4 = __float_as_uint(acc[v_4 * 8 + 4]);
            unsigned _stv8_3_5 = __float_as_uint(acc[v_4 * 8 + 5]);
            unsigned _stv8_3_6 = __float_as_uint(acc[v_4 * 8 + 6]);
            unsigned _stv8_3_7 = __float_as_uint(acc[v_4 * 8 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(partial + (pbase + (unsigned long long)pcolv))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
        }
    }
    __syncthreads();
    if (tid == 0) {
        unsigned int nb = 1;
        if (bid == 0) {
            unsigned int flip_bit = 2147483648;
            nb = flip_bit - (unsigned int)(n_chunks - 1);
        }
        unsigned int _atomic_fetch_add_4;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(_atomic_fetch_add_4) : "l"(&counters[0]), "r"(static_cast<uint32_t>(nb)) : "memory");
        flip_old[0] = _atomic_fetch_add_4;
    }
    if (tid == 0) {
        unsigned int flip_bit_1 = 2147483648;
        unsigned int flipped = (flip_old[0] ^ flip_bit_1) & flip_bit_1;
        {
            unsigned int _poll_v_5;
            do {
                asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_poll_v_5) : "l"((reinterpret_cast<unsigned int*>(counters) + (0))) : "memory");
            } while (((_poll_v_5 ^ static_cast<unsigned int>(flipped)) & static_cast<unsigned int>(flip_bit_1)) != 0);
        }
        asm volatile("fence.acquire.gpu;" ::: "memory");
    }
    __syncthreads();
    #pragma unroll 1
    for (int s_1 = bid; s_1 < 192; s_1 += n_chunks) {
        int rsub = tid / 8;
        int cg = tid - rsub * 8;
        int col = s_1 * 32 + cg * 4;
        float a[4];
        float c_0[4];
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            a[e_2] = 0.0f;
            c_0[e_2] = 0.0f;
        }
        int n_trips = (n_chunks + 47) / 48;
        #pragma unroll 4
        for (int k_1 = 0; k_1 < n_trips; k_1++) {
            int cc = rsub + k_1 * 48;
            if (cc < n_chunks) {
                float _vec_load_0[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)cc * 6144 + (unsigned long long)col);
                    _vec_load_0[0 + 0] = _v4.x;
                    _vec_load_0[0 + 1] = _v4.y;
                    _vec_load_0[0 + 2] = _v4.z;
                    _vec_load_0[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int e_3 = 0; e_3 < 4; e_3++) {
                    float y = __fmaf_rn(_vec_load_0[e_3], 1.0f, -c_0[e_3]);
                    float t = a[e_3] + y;
                    c_0[e_3] = t - a[e_3] - y;
                    a[e_3] = t;
                }
            }
        }
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            red_fold[rsub * 32 + cg * 4 + e_4] = a[e_4] - c_0[e_4];
        }
        __syncthreads();
        if (tid < 64) {
            int gi = tid / 8;
            int lane_0 = tid - gi * 8;
            float o[4];
            float oc[4];
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                o[e_5] = 0.0f;
                oc[e_5] = 0.0f;
            }
            #pragma unroll
            for (int k_2 = 0; k_2 < 6; k_2++) {
                #pragma unroll
                for (int e_6 = 0; e_6 < 4; e_6++) {
                    float d_1 = red_fold[(gi * 6 + k_2) * 32 + lane_0 * 4 + e_6];
                    float y_1 = __fmaf_rn(d_1, 1.0f, -oc[e_6]);
                    float t_1 = o[e_6] + y_1;
                    oc[e_6] = t_1 - o[e_6] - y_1;
                    o[e_6] = t_1;
                }
            }
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                grp_fold[gi * 32 + lane_0 * 4 + e_7] = o[e_7] - oc[e_7];
            }
        }
        __syncthreads();
        if (tid < 8) {
            float o2[4];
            float oc2[4];
            #pragma unroll
            for (int e_8 = 0; e_8 < 4; e_8++) {
                o2[e_8] = 0.0f;
                oc2[e_8] = 0.0f;
            }
            #pragma unroll
            for (int gk = 0; gk < 8; gk++) {
                #pragma unroll
                for (int e_9 = 0; e_9 < 4; e_9++) {
                    float d2 = grp_fold[gk * 32 + tid * 4 + e_9];
                    float y_2 = __fmaf_rn(d2, 1.0f, -oc2[e_9]);
                    float t_2 = o2[e_9] + y_2;
                    oc2[e_9] = t_2 - o2[e_9] - y_2;
                    o2[e_9] = t_2;
                }
            }
            #pragma unroll
            for (int e_10 = 0; e_10 < 4; e_10++) {
                o2[e_10] = o2[e_10] - oc2[e_10];
            }
            int ocol2 = s_1 * 32 + tid * 4;
            {
                float4 _v4 = make_float4(o2[0 + 0], o2[0 + 1], o2[0 + 2], o2[0 + 3]);
                *reinterpret_cast<float4*>(dw + (unsigned long long)ocol2) = _v4;
            }
        }
        __syncthreads();
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
