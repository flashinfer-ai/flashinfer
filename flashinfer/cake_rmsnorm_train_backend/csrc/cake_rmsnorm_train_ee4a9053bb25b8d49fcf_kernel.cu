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
#define SMEM_ROW_FOLD_STAGE_BYTES 32768
#define SMEM_ROW_FOLD_STRIDE 32768
#define SMEM_RED_FOLD_OFF 33792
#define SMEM_RED_FOLD_STAGE_BYTES 8192
#define SMEM_RED_FOLD_STRIDE 8192
#define SMEM_DOT_SLOTS_OFF 99328
#define SMEM_DOT_SLOTS_STAGE_BYTES 128
#define SMEM_DOT_SLOTS_STRIDE 128
#define SMEM_PART_SM_OFF 99456
#define SMEM_PART_SM_STAGE_BYTES 16
#define SMEM_PART_SM_STRIDE 16
#define SMEM_DESC_OFF 132352
#define SMEM_DESC_STAGE_BYTES 60
#define SMEM_DESC_STRIDE 60
#define SMEM_RTAB_OFF 132480
#define SMEM_RTAB_STAGE_BYTES 128
#define SMEM_RTAB_STRIDE 128
#define SMEM_FOLD_SM_OFF 99584
#define SMEM_FOLD_SM_STAGE_BYTES 32768
#define SMEM_FOLD_SM_STRIDE 32768
#define SMEM_TOTAL 132608
#define THREADS 512
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_train_ee4a9053bb25b8d49fcf(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define bars_addr (mbar_base + 0)
    #define dbar_addr (mbar_base + 24)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int stage_buf_addr = smem + 1024;
    float* row_fold = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_fold_addr = smem + 1024;
    float* red_fold = reinterpret_cast<float*>(smem_raw + 33792);
    const int red_fold_addr = smem + 33792;
    float* dot_slots = reinterpret_cast<float*>(smem_raw + 99328);
    const int dot_slots_addr = smem + 99328;
    float* part_sm = reinterpret_cast<float*>(smem_raw + 99456);
    const int part_sm_addr = smem + 99456;
    int* desc = reinterpret_cast<int*>(smem_raw + 132352);
    const int desc_addr = smem + 132352;
    int* rtab = reinterpret_cast<int*>(smem_raw + 132480);
    const int rtab_addr = smem + 132480;
    float* fold_sm = reinterpret_cast<float*>(smem_raw + 99584);
    const int fold_sm_addr = smem + 99584;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // bars: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // dbar: 5 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int c = bid;
    int rg = tid / 256;
    int tir = tid - rg * 256;
    unsigned int claim[1];
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        claim[0] = _atomic_fetch_add_0;
        int S = (T + 1) / 2;
        int _min_1 = ((rows_per_chunk) < (S) ? (rows_per_chunk) : (S));
        int sl = ((_min_1) > (1) ? (_min_1) : (1));
        int share = (S + sl - 1) / sl;
        int tail = (((share + 11) / 12) > (1) ? ((share + 11) / 12) : (1));
        int size = (share + 1) / 2;
        int rem = S;
        int base = 0;
        int first = 0;
        int nr = 0;
        int walk = 1;
        while (walk == 1) {
            walk = 0;
            if (nr < 5) {
                if (size > tail) {
                    if (rem >= sl * size) {
                        rtab[nr * 3] = first;
                        rtab[nr * 3 + 1] = base;
                        rtab[nr * 3 + 2] = size;
                        first += sl;
                        base += sl * size;
                        rem -= sl * size;
                        size = (size + 1) / 2;
                        nr += 1;
                        walk = 1;
                    }
                }
            }
        }
        int q = rem / sl;
        int m = rem - q * sl;
        if (m > 0) {
            rtab[nr * 3] = first;
            rtab[nr * 3 + 1] = base;
            rtab[nr * 3 + 2] = q + 1;
            first += m;
            base += m * (q + 1);
            nr += 1;
        }
        if (q > 0) {
            rtab[nr * 3] = first;
            rtab[nr * 3 + 1] = base;
            rtab[nr * 3 + 2] = q;
            first += sl - m;
            base += (sl - m) * q;
            nr += 1;
        }
        rtab[nr * 3] = first;
        rtab[nr * 3 + 1] = base;
        rtab[nr * 3 + 2] = 0;
        rtab[31] = nr + 1;
    }
    int ring = (int)stage_buf_addr;
    int my_g = ring + rg * 8192 + tir * 16;
    unsigned int ww[8];
    #pragma unroll
    for (int v = 0; v < 2; v++) {
        int wcol = (v * 256 + tir) * 8;
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)wcol);
            uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[v * 4]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_2[_blk] = _vptr_2[_blk];
            }
        }
    }
    int rng[2];
    int nd[3];
    if (tid == 0) {
        int cid0 = (int)claim[0];
        int nr_1 = rtab[31];
        int r_0 = 0;
        int walk_1 = 1;
        while (walk_1 == 1) {
            walk_1 = 0;
            if (nr_1 > r_0 + 2) {
                if (cid0 >= rtab[(r_0 + 1) * 3]) {
                    r_0 += 1;
                    walk_1 = 1;
                }
            }
        }
        if (cid0 >= rtab[(nr_1 - 1) * 3]) {
            rng[0] = T;
            rng[1] = T;
        } else {
            int first_1 = rtab[r_0 * 3];
            int base_1 = rtab[r_0 * 3 + 1];
            int stages = rtab[r_0 * 3 + 2];
            rng[0] = (base_1 + (cid0 - first_1) * stages) * 2;
            int _min_3 = ((rng[0] + stages * 2) < (T) ? (rng[0] + stages * 2) : (T));
            rng[1] = _min_3;
        }
        int entry = 0;
        int word = entry * 3;
        desc[word] = cid0;
        desc[word + 1] = rng[0];
        desc[word + 2] = rng[1];
        mbarrier_arrive(dbar_addr + (entry) * 8);
        if (rng[0] < T) {
            unsigned int _atomic_fetch_add_4 = atomicAdd(&counters[0], 1);
            claim[0] = _atomic_fetch_add_4;
        } else {
            int nr_0 = rtab[31];
            int last = rtab[(nr_0 - 1) * 3] + num_bids - 1;
            if (T == 0) {
                last = 0;
            }
            if (cid0 == last) {
                *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
            }
        }
    }
    int entry_1 = 0;
    int use = 0;
    mbarrier_wait(dbar_addr + (entry_1) * 8, use);
    int word_1 = entry_1 * 3;
    int w0 = desc[word_1];
    int w1 = desc[word_1 + 1];
    int w2 = desc[word_1 + 2];
    int _warp_redux_5;
    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_5) : "r"(w0));
    nd[0] = _warp_redux_5;
    int _warp_redux_6;
    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_6) : "r"(w1));
    nd[1] = _warp_redux_6;
    int _warp_redux_7;
    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_7) : "r"(w2));
    nd[2] = _warp_redux_7;
    int c0 = nd[0];
    int cb = nd[1];
    int ce = nd[2];
    int ps[3];
    ps[0] = nd[1];
    ps[1] = nd[2];
    ps[2] = 0;
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        if (ps[0] < ps[1]) {
            int first_2 = ps[0];
            int rows_k = ((ps[1] - first_2) < (2) ? (ps[1] - first_2) : (2));
            int sbase = ring + s * 32768;
            if (tid == 0) {
                mbarrier_arrive_expect_tx(bars_addr + (s) * 8, rows_k * 16384);
            }
            if (tir == 0) {
                if (rg < rows_k) {
                    int row_g = first_2 + rg;
                    long long g_srcg = (long long)row_g * g_stride;
                    long long x_srcg = (long long)row_g * x_stride;
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sbase + rg * 8192), "l"(reinterpret_cast<const uint8_t*>(g + g_srcg)), "r"((unsigned)(8192)), "r"(bars_addr + (s) * 8),
                           "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1], %2, [%3], %4;"
                        :: "r"(sbase + (rg + 2) * 8192), "l"(reinterpret_cast<const uint8_t*>(x + x_srcg)), "r"((unsigned)(8192)), "r"(bars_addr + (s) * 8),
                           "l"(0x14F0000000000000ULL) : "memory");
                }
            }
            ps[0] = ps[0] + 2;
            if (ps[0] >= ps[1]) {
                ps[2] = ps[2] + 1;
                if (tid == 0) {
                    int cid = (int)claim[0];
                    int nr_2 = rtab[31];
                    int r_0_1 = 0;
                    int walk_2 = 1;
                    while (walk_2 == 1) {
                        walk_2 = 0;
                        if (nr_2 > r_0_1 + 2) {
                            if (cid >= rtab[(r_0_1 + 1) * 3]) {
                                r_0_1 += 1;
                                walk_2 = 1;
                            }
                        }
                    }
                    if (cid >= rtab[(nr_2 - 1) * 3]) {
                        rng[0] = T;
                        rng[1] = T;
                    } else {
                        int first_0 = rtab[r_0_1 * 3];
                        int base_2 = rtab[r_0_1 * 3 + 1];
                        int stages_1 = rtab[r_0_1 * 3 + 2];
                        rng[0] = (base_2 + (cid - first_0) * stages_1) * 2;
                        int _min_8 = ((rng[0] + stages_1 * 2) < (T) ? (rng[0] + stages_1 * 2) : (T));
                        rng[1] = _min_8;
                    }
                    int entry_1_1 = ps[2] % 5;
                    int word_2 = entry_1_1 * 3;
                    desc[word_2] = cid;
                    desc[word_2 + 1] = rng[0];
                    desc[word_2 + 2] = rng[1];
                    mbarrier_arrive(dbar_addr + (entry_1_1) * 8);
                    if (rng[0] < T) {
                        unsigned int _atomic_fetch_add_9 = atomicAdd(&counters[0], 1);
                        claim[0] = _atomic_fetch_add_9;
                    } else {
                        int nr_0_1 = rtab[31];
                        int last_1 = rtab[(nr_0_1 - 1) * 3] + num_bids - 1;
                        if (T == 0) {
                            last_1 = 0;
                        }
                        if (cid == last_1) {
                            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
                        }
                    }
                }
                int entry_0 = ps[2] % 5;
                int use_1 = ps[2] / 5 & 1;
                mbarrier_wait(dbar_addr + (entry_0) * 8, use_1);
                int word_2_1 = entry_0 * 3;
                int w0_3 = desc[word_2_1];
                int w1_4 = desc[word_2_1 + 1];
                int w2_5 = desc[word_2_1 + 2];
                int _warp_redux_10;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_10) : "r"(w0_3));
                nd[0] = _warp_redux_10;
                int _warp_redux_11;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_11) : "r"(w1_4));
                nd[1] = _warp_redux_11;
                int _warp_redux_12;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_12) : "r"(w2_5));
                nd[2] = _warp_redux_12;
                ps[0] = nd[1];
                ps[1] = nd[2];
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
    int n_cons = 0;
    int g_step = 0;
    float r_cur = 0.0f;
    if (cb < ce) {
        int row0 = ((cb + rg) < (ce - 1) ? (cb + rg) : (ce - 1));
        r_cur = r[row0];
    }
    unsigned int gw[8];
    unsigned int xw[8];
    float out[16];
    if (cb >= ce) {
        if (c0 < n_chunks) {
            unsigned long long pbase0 = (unsigned long long)c0 * 4096;
            float fold0[8];
            #pragma unroll
            for (int e0 = 0; e0 < 8; e0++) {
                fold0[e0] = 0.0f;
            }
            int pcol0 = tid * 8;
            {
                unsigned _stv8_13_0 = __float_as_uint(fold0[0 + 0]);
                unsigned _stv8_13_1 = __float_as_uint(fold0[0 + 1]);
                unsigned _stv8_13_2 = __float_as_uint(fold0[0 + 2]);
                unsigned _stv8_13_3 = __float_as_uint(fold0[0 + 3]);
                unsigned _stv8_13_4 = __float_as_uint(fold0[0 + 4]);
                unsigned _stv8_13_5 = __float_as_uint(fold0[0 + 5]);
                unsigned _stv8_13_6 = __float_as_uint(fold0[0 + 6]);
                unsigned _stv8_13_7 = __float_as_uint(fold0[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(partial + (pbase0 + (unsigned long long)pcol0))), "r"(_stv8_13_0), "r"(_stv8_13_1), "r"(_stv8_13_2), "r"(_stv8_13_3), "r"(_stv8_13_4), "r"(_stv8_13_5), "r"(_stv8_13_6), "r"(_stv8_13_7) : "memory");
            }
        }
    }
    while (cb < ce) {
        int n_steps_c = (ce - cb + 1) / 2;
        #pragma unroll 1
        for (int k = 0; k < n_steps_c; k++) {
            int slot = g_step % 3;
            int parity = g_step / 3 & 1;
            int par = (g_step & 1) * 16;
            int row = cb + k * 2 + rg;
            int row_n = ((row + 2) < (ce - 1) ? (row + 2) : (ce - 1));
            if (n_steps_c <= k + 1) {
                int entry_0_1 = (n_cons + 1) % 5;
                int use_1_1 = (n_cons + 1) / 5 & 1;
                mbarrier_wait(dbar_addr + (entry_0_1) * 8, use_1_1);
                int word_2_2 = entry_0_1 * 3;
                int w0_3_1 = desc[word_2_2];
                int w1_4_1 = desc[word_2_2 + 1];
                int w2_5_1 = desc[word_2_2 + 2];
                int _warp_redux_14;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_14) : "r"(w0_3_1));
                nd[0] = _warp_redux_14;
                int _warp_redux_15;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_15) : "r"(w1_4_1));
                nd[1] = _warp_redux_15;
                int _warp_redux_16;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_16) : "r"(w2_5_1));
                nd[2] = _warp_redux_16;
                row_n = cb;
                if (nd[1] < T) {
                    int _min_17 = ((nd[1] + rg) < (nd[2] - 1) ? (nd[1] + rg) : (nd[2] - 1));
                    row_n = _min_17;
                }
            }
            float r_nxt = r[row_n];
            mbarrier_wait(bars_addr + (slot) * 8, parity);
            int sg = my_g + slot * 32768;
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                __int128_t _smem_raw_18;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(sg + v_1 * 4096));
                gw[v_1 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
                gw[(v_1 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
                gw[(v_1 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
                gw[(v_1 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
                __int128_t _smem_raw_19;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_19) : "r"(sg + (16384 + v_1 * 4096)));
                xw[v_1 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[0];
                xw[(v_1 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[1];
                xw[(v_1 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[2];
                xw[(v_1 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_19)[3];
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
            if (lane == 0) {
                dot_slots[par + warp] = dot;
            }
            __syncthreads();
            if (ps[0] < ps[1]) {
                int first_3 = ps[0];
                int rows_k_1 = ((ps[1] - first_3) < (2) ? (ps[1] - first_3) : (2));
                int sbase_1 = ring + slot * 32768;
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(bars_addr + (slot) * 8, rows_k_1 * 16384);
                }
                if (tir == 0) {
                    if (rg < rows_k_1) {
                        int row_g_1 = first_3 + rg;
                        long long g_srcg_1 = (long long)row_g_1 * g_stride;
                        long long x_srcg_1 = (long long)row_g_1 * x_stride;
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase_1 + rg * 8192), "l"(reinterpret_cast<const uint8_t*>(g + g_srcg_1)), "r"((unsigned)(8192)), "r"(bars_addr + (slot) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1], %2, [%3], %4;"
                            :: "r"(sbase_1 + (rg + 2) * 8192), "l"(reinterpret_cast<const uint8_t*>(x + x_srcg_1)), "r"((unsigned)(8192)), "r"(bars_addr + (slot) * 8),
                               "l"(0x14F0000000000000ULL) : "memory");
                    }
                }
                ps[0] = ps[0] + 2;
                if (ps[0] >= ps[1]) {
                    ps[2] = ps[2] + 1;
                    if (tid == 0) {
                        int cid_1 = (int)claim[0];
                        int nr_3 = rtab[31];
                        int r_0_2 = 0;
                        int walk_3 = 1;
                        while (walk_3 == 1) {
                            walk_3 = 0;
                            if (nr_3 > r_0_2 + 2) {
                                if (cid_1 >= rtab[(r_0_2 + 1) * 3]) {
                                    r_0_2 += 1;
                                    walk_3 = 1;
                                }
                            }
                        }
                        if (cid_1 >= rtab[(nr_3 - 1) * 3]) {
                            rng[0] = T;
                            rng[1] = T;
                        } else {
                            int first_0_1 = rtab[r_0_2 * 3];
                            int base_3 = rtab[r_0_2 * 3 + 1];
                            int stages_2 = rtab[r_0_2 * 3 + 2];
                            rng[0] = (base_3 + (cid_1 - first_0_1) * stages_2) * 2;
                            int _min_20 = ((rng[0] + stages_2 * 2) < (T) ? (rng[0] + stages_2 * 2) : (T));
                            rng[1] = _min_20;
                        }
                        int entry_1_2 = ps[2] % 5;
                        int word_2_3 = entry_1_2 * 3;
                        desc[word_2_3] = cid_1;
                        desc[word_2_3 + 1] = rng[0];
                        desc[word_2_3 + 2] = rng[1];
                        mbarrier_arrive(dbar_addr + (entry_1_2) * 8);
                        if (rng[0] < T) {
                            unsigned int _atomic_fetch_add_21 = atomicAdd(&counters[0], 1);
                            claim[0] = _atomic_fetch_add_21;
                        } else {
                            int nr_0_2 = rtab[31];
                            int last_2 = rtab[(nr_0_2 - 1) * 3] + num_bids - 1;
                            if (T == 0) {
                                last_2 = 0;
                            }
                            if (cid_1 == last_2) {
                                *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
                            }
                        }
                    }
                    int entry_0_2 = ps[2] % 5;
                    int use_1_2 = ps[2] / 5 & 1;
                    mbarrier_wait(dbar_addr + (entry_0_2) * 8, use_1_2);
                    int word_2_4 = entry_0_2 * 3;
                    int w0_3_2 = desc[word_2_4];
                    int w1_4_2 = desc[word_2_4 + 1];
                    int w2_5_2 = desc[word_2_4 + 2];
                    int _warp_redux_22;
                    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_22) : "r"(w0_3_2));
                    nd[0] = _warp_redux_22;
                    int _warp_redux_23;
                    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_23) : "r"(w1_4_2));
                    nd[1] = _warp_redux_23;
                    int _warp_redux_24;
                    asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_24) : "r"(w2_5_2));
                    nd[2] = _warp_redux_24;
                    ps[0] = nd[1];
                    ps[1] = nd[2];
                }
            }
            float tot = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < 8; kk++) {
                float d = dot_slots[par + rg * 8 + kk];
                tot += d;
            }
            float rq = r_cur;
            float cq = __fdiv_rn(tot, 4096.0f);
            float kx = rq * rq * rq * cq;
            float nkx = -kx;
            if (row < ce) {
                unsigned long long dx_base = (unsigned long long)row * 4096;
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
                    int dcol = (v_3 * 256 + tir) * 8;
                    unsigned int _packed_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((out + v_3 * 8)[_lp*2 + 0], (out + v_3 * 8)[_lp*2+1 + 0]));
                        _packed_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((out + v_3 * 8)[_lp*2 + 2], (out + v_3 * 8)[_lp*2+1 + 2]));
                        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((out + v_3 * 8)[_lp*2 + 4], (out + v_3 * 8)[_lp*2+1 + 4]));
                        _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((out + v_3 * 8)[_lp*2 + 6], (out + v_3 * 8)[_lp*2+1 + 6]));
                        _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(dx + (dx_base + (unsigned long long)dcol))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                }
            }
            r_cur = r_nxt;
            g_step += 1;
        }
        int entry_0_3 = n_cons % 5;
        int use_1_3 = n_cons / 5 & 1;
        mbarrier_wait(dbar_addr + (entry_0_3) * 8, use_1_3);
        int word_2_5 = entry_0_3 * 3;
        int w0_3_3 = desc[word_2_5];
        int w1_4_3 = desc[word_2_5 + 1];
        int w2_5_3 = desc[word_2_5 + 2];
        int _warp_redux_25;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_25) : "r"(w0_3_3));
        nd[0] = _warp_redux_25;
        int _warp_redux_26;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_26) : "r"(w1_4_3));
        nd[1] = _warp_redux_26;
        int _warp_redux_27;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_27) : "r"(w2_5_3));
        nd[2] = _warp_redux_27;
        int cid_f = nd[0];
        unsigned long long pbase = (unsigned long long)cid_f * 4096;
        #pragma unroll
        for (int v_4 = 0; v_4 < 2; v_4++) {
            int fcol = (v_4 * 256 + tir) * 8;
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2++) {
                fold_sm[rg * 4096 + fcol + e_2] = acc[v_4 * 8 + e_2];
            }
        }
        __syncthreads();
        float fold_acc[8];
        #pragma unroll
        for (int e_3 = 0; e_3 < 8; e_3++) {
            fold_acc[e_3] = 0.0f;
        }
        #pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            #pragma unroll
            for (int e_4 = 0; e_4 < 8; e_4++) {
                float d2 = fold_sm[k2 * 4096 + tid * 8 + e_4];
                fold_acc[e_4] = fold_acc[e_4] + d2;
            }
        }
        int pcol = tid * 8;
        {
            unsigned _stv8_28_0 = __float_as_uint(fold_acc[0 + 0]);
            unsigned _stv8_28_1 = __float_as_uint(fold_acc[0 + 1]);
            unsigned _stv8_28_2 = __float_as_uint(fold_acc[0 + 2]);
            unsigned _stv8_28_3 = __float_as_uint(fold_acc[0 + 3]);
            unsigned _stv8_28_4 = __float_as_uint(fold_acc[0 + 4]);
            unsigned _stv8_28_5 = __float_as_uint(fold_acc[0 + 5]);
            unsigned _stv8_28_6 = __float_as_uint(fold_acc[0 + 6]);
            unsigned _stv8_28_7 = __float_as_uint(fold_acc[0 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(partial + (pbase + (unsigned long long)pcol))), "r"(_stv8_28_0), "r"(_stv8_28_1), "r"(_stv8_28_2), "r"(_stv8_28_3), "r"(_stv8_28_4), "r"(_stv8_28_5), "r"(_stv8_28_6), "r"(_stv8_28_7) : "memory");
        }
        #pragma unroll
        for (int j2 = 0; j2 < 16; j2++) {
            acc[j2] = 0.0f;
        }
        int entry_6 = (n_cons + 1) % 5;
        int use_7 = (n_cons + 1) / 5 & 1;
        mbarrier_wait(dbar_addr + (entry_6) * 8, use_7);
        int word_8 = entry_6 * 3;
        int w0_9 = desc[word_8];
        int w1_10 = desc[word_8 + 1];
        int w2_11 = desc[word_8 + 2];
        int _warp_redux_29;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_29) : "r"(w0_9));
        nd[0] = _warp_redux_29;
        int _warp_redux_30;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_30) : "r"(w1_10));
        nd[1] = _warp_redux_30;
        int _warp_redux_31;
        asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_31) : "r"(w2_11));
        nd[2] = _warp_redux_31;
        cb = nd[1];
        ce = nd[2];
        n_cons += 1;
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
