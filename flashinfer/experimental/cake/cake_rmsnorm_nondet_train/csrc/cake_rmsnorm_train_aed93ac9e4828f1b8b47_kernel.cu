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

#define LAUNCH_MIN_BLOCKS 2

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
kernel_cake_rmsnorm_train_aed93ac9e4828f1b8b47(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    double* sums = reinterpret_cast<double*>(smem_raw + 1120);
    const int sums_addr = smem + 1120;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1120);
    const int dwbuf_addr = smem + 1120;
    float* pairs = reinterpret_cast<float*>(smem_raw + 1120);
    const int pairs_addr = smem + 1120;
    int* meta = reinterpret_cast<int*>(smem_raw + 1088);
    const int meta_addr = smem + 1088;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 33920);
    const int ring_addr = smem + 33920;

    // Mbarrier init (3 barriers at smem_raw[0..24))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int nxt = 0;
    if (tid == 0) {
        unsigned int _atomic_fetch_add_0 = atomicAdd(&counters[0], 1);
        nxt = _atomic_fetch_add_0;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long g_stride = g_stride_p0;
    long long x_stride = x_stride_p0;
    long long gh_stride = g_h_stride_p0;
    int rg = warp / 4;
    int tir = tid - rg * 128;
    unsigned int ww[16];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((128 + tir) * 8));
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(w + (unsigned long long)((256 + tir) * 8));
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&ww[8]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_3[_blk] = _vptr_3[_blk];
        }
    }
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(w + (unsigned long long)((384 + tir) * 8));
        uint4* _vdst_4 = reinterpret_cast<uint4*>(&ww[12]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_4[_blk] = _vptr_4[_blk];
        }
    }
    float acc[32];
    float comp[32];
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
    acc[16] = 0.0f;
    comp[16] = 0.0f;
    acc[17] = 0.0f;
    comp[17] = 0.0f;
    acc[18] = 0.0f;
    comp[18] = 0.0f;
    acc[19] = 0.0f;
    comp[19] = 0.0f;
    acc[20] = 0.0f;
    comp[20] = 0.0f;
    acc[21] = 0.0f;
    comp[21] = 0.0f;
    acc[22] = 0.0f;
    comp[22] = 0.0f;
    acc[23] = 0.0f;
    comp[23] = 0.0f;
    acc[24] = 0.0f;
    comp[24] = 0.0f;
    acc[25] = 0.0f;
    comp[25] = 0.0f;
    acc[26] = 0.0f;
    comp[26] = 0.0f;
    acc[27] = 0.0f;
    comp[27] = 0.0f;
    acc[28] = 0.0f;
    comp[28] = 0.0f;
    acc[29] = 0.0f;
    comp[29] = 0.0f;
    acc[30] = 0.0f;
    comp[30] = 0.0f;
    acc[31] = 0.0f;
    comp[31] = 0.0f;
    int c = 0;
    int first_c = 0;
    int rows_mine = 0;
    int slot = 0;
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[16];
    unsigned int xw[16];
    unsigned int hw[1];
    float rr[1];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        pc = (int)nxt;
        first_c = pc;
        if (pc < n_chunks) {
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
            for (int k = 0; k < 5; k++) {
                if (halving != 0) {
                    if (size > tail) {
                        if (rem >= sl * size) {
                            if (pc < first + sl) {
                                span = size;
                                begin_it = base + (pc - first) * size;
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
                if (pc < first + extra) {
                    span = quot + 1;
                    begin_it = base + (pc - first) * (quot + 1);
                } else if (pc < first + sl) {
                    span = quot;
                    begin_it = base + extra * (quot + 1) + (pc - first - extra) * quot;
                } else {
                    begin_it = stages;
                }
            }
            int _min_5 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_5;
            int _min_6 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_6;
            unsigned int _atomic_fetch_add_7 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_7;
            unsigned int _atomic_fetch_add_8 = atomicAdd(&counters[3], 1);
            slot = (int)_atomic_fetch_add_8;
        }
        int slot_0 = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
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
                    for (int k_1 = 0; k_1 < 5; k_1++) {
                        if (halving_1 != 0) {
                            if (size_1 > tail_1) {
                                if (rem_1 >= sl_1 * size_1) {
                                    if (pc < first_1 + sl_1) {
                                        span_1 = size_1;
                                        begin_it_1 = base_1 + (pc - first_1) * size_1;
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
                        if (pc < first_1 + extra_1) {
                            span_1 = quot_1 + 1;
                            begin_it_1 = base_1 + (pc - first_1) * (quot_1 + 1);
                        } else if (pc < first_1 + sl_1) {
                            span_1 = quot_1;
                            begin_it_1 = base_1 + extra_1 * (quot_1 + 1) + (pc - first_1 - extra_1) * quot_1;
                        } else {
                            begin_it_1 = stages_1;
                        }
                    }
                    int _min_9 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_9;
                    int _min_10 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_10;
                    unsigned int _atomic_fetch_add_11 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_11;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_12 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_12;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 16384);
            int sbase = ring_at + slot_0 * 16384;
            int _min_13 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_13);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + row_i_0 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_0 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval;
        int slot_1 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_2 = T;
                    int sl_2 = rows_per_chunk;
                    if (sl_2 > stages_2) {
                        sl_2 = stages_2;
                    }
                    if (sl_2 < 1) {
                        sl_2 = 1;
                    }
                    int share_2 = (stages_2 + sl_2 - 1) / sl_2;
                    int tail_2 = (share_2 + 11) / 12;
                    if (tail_2 < 1) {
                        tail_2 = 1;
                    }
                    int rem_2 = stages_2;
                    int size_2 = (share_2 + 1) / 2;
                    int first_2 = 0;
                    int base_2 = 0;
                    int span_2 = 0;
                    int halving_2 = 1;
                    int begin_it_2 = 0;
                    #pragma unroll 1
                    for (int k_2 = 0; k_2 < 5; k_2++) {
                        if (halving_2 != 0) {
                            if (size_2 > tail_2) {
                                if (rem_2 >= sl_2 * size_2) {
                                    if (pc < first_2 + sl_2) {
                                        span_2 = size_2;
                                        begin_it_2 = base_2 + (pc - first_2) * size_2;
                                        halving_2 = 0;
                                    } else {
                                        first_2 = first_2 + sl_2;
                                        base_2 = base_2 + sl_2 * size_2;
                                        rem_2 = rem_2 - sl_2 * size_2;
                                        size_2 = (size_2 + 1) / 2;
                                    }
                                } else {
                                    halving_2 = 0;
                                }
                            } else {
                                halving_2 = 0;
                            }
                        }
                    }
                    if (span_2 == 0) {
                        int quot_2 = rem_2 / sl_2;
                        int extra_2 = rem_2 - quot_2 * sl_2;
                        if (pc < first_2 + extra_2) {
                            span_2 = quot_2 + 1;
                            begin_it_2 = base_2 + (pc - first_2) * (quot_2 + 1);
                        } else if (pc < first_2 + sl_2) {
                            span_2 = quot_2;
                            begin_it_2 = base_2 + extra_2 * (quot_2 + 1) + (pc - first_2 - extra_2) * quot_2;
                        } else {
                            begin_it_2 = stages_2;
                        }
                    }
                    int _min_14 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_14;
                    int _min_15 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_15;
                    unsigned int _atomic_fetch_add_16 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_16;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_17 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_17;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 16384);
            int sbase_1 = ring_at + slot_1 * 16384;
            int _min_18 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_18);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + row_i_1 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_1 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + 8),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_0;
        int slot_2 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_3 = T;
                    int sl_3 = rows_per_chunk;
                    if (sl_3 > stages_3) {
                        sl_3 = stages_3;
                    }
                    if (sl_3 < 1) {
                        sl_3 = 1;
                    }
                    int share_3 = (stages_3 + sl_3 - 1) / sl_3;
                    int tail_3 = (share_3 + 11) / 12;
                    if (tail_3 < 1) {
                        tail_3 = 1;
                    }
                    int rem_3 = stages_3;
                    int size_3 = (share_3 + 1) / 2;
                    int first_3 = 0;
                    int base_3 = 0;
                    int span_3 = 0;
                    int halving_3 = 1;
                    int begin_it_3 = 0;
                    #pragma unroll 1
                    for (int k_3 = 0; k_3 < 5; k_3++) {
                        if (halving_3 != 0) {
                            if (size_3 > tail_3) {
                                if (rem_3 >= sl_3 * size_3) {
                                    if (pc < first_3 + sl_3) {
                                        span_3 = size_3;
                                        begin_it_3 = base_3 + (pc - first_3) * size_3;
                                        halving_3 = 0;
                                    } else {
                                        first_3 = first_3 + sl_3;
                                        base_3 = base_3 + sl_3 * size_3;
                                        rem_3 = rem_3 - sl_3 * size_3;
                                        size_3 = (size_3 + 1) / 2;
                                    }
                                } else {
                                    halving_3 = 0;
                                }
                            } else {
                                halving_3 = 0;
                            }
                        }
                    }
                    if (span_3 == 0) {
                        int quot_3 = rem_3 / sl_3;
                        int extra_3 = rem_3 - quot_3 * sl_3;
                        if (pc < first_3 + extra_3) {
                            span_3 = quot_3 + 1;
                            begin_it_3 = base_3 + (pc - first_3) * (quot_3 + 1);
                        } else if (pc < first_3 + sl_3) {
                            span_3 = quot_3;
                            begin_it_3 = base_3 + extra_3 * (quot_3 + 1) + (pc - first_3 - extra_3) * quot_3;
                        } else {
                            begin_it_3 = stages_3;
                        }
                    }
                    int _min_19 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_19;
                    int _min_20 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_20;
                    unsigned int _atomic_fetch_add_21 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_21;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_22 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_22;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 16384);
            int sbase_2 = ring_at + slot_2 * 16384;
            int _min_23 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_23);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + row_i_2 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_2 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + 16),
                   "l"(0x14F0000000000000ULL) : "memory");
        }
        prow = prow + nval_1;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_2 = meta[1];
    rr[0] = 0.0f;
    if (nval_2 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_4 = 0; k_4 < T + n_chunks; k_4++) {
        if (nval_2 == 0) {
            break;
        }
        rows_mine = rows_mine + nval_2;
        int _stage_full = (int)((unsigned int)k_4 % 3);
        int _parity_full = (int)((unsigned int)k_4 / 3 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_4 % 3 * 16384 + rg * 8192;
        __int128_t _smem_raw_24;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_24) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_24)[3];
        __int128_t _smem_raw_25;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_25) : "r"(rbase + 8192 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_25)[3];
        __int128_t _smem_raw_26;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_26) : "r"(rbase + (128 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[3];
        __int128_t _smem_raw_27;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_27) : "r"(rbase + 8192 + (128 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[3];
        __int128_t _smem_raw_28;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_28) : "r"(rbase + (256 + tir) * 8 * 2));
        gw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[0];
        gw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[1];
        gw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[2];
        gw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[3];
        __int128_t _smem_raw_29;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_29) : "r"(rbase + 8192 + (256 + tir) * 8 * 2));
        xw[8] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[0];
        xw[9] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[1];
        xw[10] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[2];
        xw[11] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[3];
        __int128_t _smem_raw_30;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_30) : "r"(rbase + (384 + tir) * 8 * 2));
        gw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[0];
        gw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[1];
        gw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[2];
        gw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[3];
        __int128_t _smem_raw_31;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_31) : "r"(rbase + 8192 + (384 + tir) * 8 * 2));
        xw[12] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[0];
        xw[13] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[1];
        xw[14] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[2];
        xw[15] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[3];
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
        dq = __fmaf_rn(__uint_as_float(gw[8] << 16) * __uint_as_float(ww[8] << 16), __uint_as_float(xw[8] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[8] & 4294901760u) * __uint_as_float(ww[8] & 4294901760u), __uint_as_float(xw[8] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] << 16) * __uint_as_float(ww[9] << 16), __uint_as_float(xw[9] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[9] & 4294901760u) * __uint_as_float(ww[9] & 4294901760u), __uint_as_float(xw[9] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] << 16) * __uint_as_float(ww[10] << 16), __uint_as_float(xw[10] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[10] & 4294901760u) * __uint_as_float(ww[10] & 4294901760u), __uint_as_float(xw[10] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] << 16) * __uint_as_float(ww[11] << 16), __uint_as_float(xw[11] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[11] & 4294901760u) * __uint_as_float(ww[11] & 4294901760u), __uint_as_float(xw[11] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] << 16) * __uint_as_float(ww[12] << 16), __uint_as_float(xw[12] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[12] & 4294901760u) * __uint_as_float(ww[12] & 4294901760u), __uint_as_float(xw[12] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] << 16) * __uint_as_float(ww[13] << 16), __uint_as_float(xw[13] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[13] & 4294901760u) * __uint_as_float(ww[13] & 4294901760u), __uint_as_float(xw[13] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] << 16) * __uint_as_float(ww[14] << 16), __uint_as_float(xw[14] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[14] & 4294901760u) * __uint_as_float(ww[14] & 4294901760u), __uint_as_float(xw[14] & 4294901760u), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] << 16) * __uint_as_float(ww[15] << 16), __uint_as_float(xw[15] << 16), dq);
        dq = __fmaf_rn(__uint_as_float(gw[15] & 4294901760u) * __uint_as_float(ww[15] & 4294901760u), __uint_as_float(xw[15] & 4294901760u), dq);
        float _warp_reduce_32 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_32 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_32, offset);
        dot[0] = _warp_reduce_32;
        if (lane == 0) {
            dots[(k_4 & 1) * 4 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_4 & 1) * 4 + rg * 4];
        tot += dots[(k_4 & 1) * 4 + rg * 4 + 1];
        tot += dots[(k_4 & 1) * 4 + rg * 4 + 2];
        tot += dots[(k_4 & 1) * 4 + rg * 4 + 3];
        dot[0] = tot;
        if (tid == 0) {
            int slot_3 = (k_4 + 3) % 3;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt;
                    if (pc < n_chunks) {
                        int stages_4 = T;
                        int sl_4 = rows_per_chunk;
                        if (sl_4 > stages_4) {
                            sl_4 = stages_4;
                        }
                        if (sl_4 < 1) {
                            sl_4 = 1;
                        }
                        int share_4 = (stages_4 + sl_4 - 1) / sl_4;
                        int tail_4 = (share_4 + 11) / 12;
                        if (tail_4 < 1) {
                            tail_4 = 1;
                        }
                        int rem_4 = stages_4;
                        int size_4 = (share_4 + 1) / 2;
                        int first_4 = 0;
                        int base_4 = 0;
                        int span_4 = 0;
                        int halving_4 = 1;
                        int begin_it_4 = 0;
                        #pragma unroll 1
                        for (int k_0 = 0; k_0 < 5; k_0++) {
                            if (halving_4 != 0) {
                                if (size_4 > tail_4) {
                                    if (rem_4 >= sl_4 * size_4) {
                                        if (pc < first_4 + sl_4) {
                                            span_4 = size_4;
                                            begin_it_4 = base_4 + (pc - first_4) * size_4;
                                            halving_4 = 0;
                                        } else {
                                            first_4 = first_4 + sl_4;
                                            base_4 = base_4 + sl_4 * size_4;
                                            rem_4 = rem_4 - sl_4 * size_4;
                                            size_4 = (size_4 + 1) / 2;
                                        }
                                    } else {
                                        halving_4 = 0;
                                    }
                                } else {
                                    halving_4 = 0;
                                }
                            }
                        }
                        if (span_4 == 0) {
                            int quot_4 = rem_4 / sl_4;
                            int extra_4 = rem_4 - quot_4 * sl_4;
                            if (pc < first_4 + extra_4) {
                                span_4 = quot_4 + 1;
                                begin_it_4 = base_4 + (pc - first_4) * (quot_4 + 1);
                            } else if (pc < first_4 + sl_4) {
                                span_4 = quot_4;
                                begin_it_4 = base_4 + extra_4 * (quot_4 + 1) + (pc - first_4 - extra_4) * quot_4;
                            } else {
                                begin_it_4 = stages_4;
                            }
                        }
                        int _min_33 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                        prow = _min_33;
                        int _min_34 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                        pend = _min_34;
                        unsigned int _atomic_fetch_add_35 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_35;
                    }
                }
            }
            int nval_2_1 = 0;
            if (prow < pend) {
                int _min_36 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_2_1 = _min_36;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_2_1;
            if (nval_2_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8, 16384);
                int sbase_3 = ring_at + slot_3 * 16384;
                int _min_37 = ((nval_2_1 - 1) < (0) ? (nval_2_1 - 1) : (0));
                long long row_i_3 = (long long)(prow + _min_37);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + row_i_3 * g_stride)), "r"((unsigned)(8192)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_3 + 8192), "l"(reinterpret_cast<const uint8_t*>(x + row_i_3 * x_stride)), "r"((unsigned)(8192)), "r"(full_addr + ((int)((unsigned int)(k_4 + 3) % 3)) * 8),
                       "l"(0x14F0000000000000ULL) : "memory");
            }
            prow = prow + nval_2_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 4096.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        float2 rr2 = make_float2(rq, rq);
        if (row0 + rg < row0 + nval_2) {
            float out_0[8];
            float g0_0 = __uint_as_float(gw[0] << 16);
            float g1_0 = __uint_as_float(gw[0] & 4294901760u);
            float x0_0 = __uint_as_float(xw[0] << 16);
            float x1_0 = __uint_as_float(xw[0] & 4294901760u);
            float2 _f32x2_mul_a_38 = make_float2(g0_0, g1_0);
            float2 _f32x2_mul_b_39 = make_float2(x0_0, x1_0);
            float2 gx;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_39));
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
            float2 _f32x2_mul_a_40 = make_float2(g0_1, g1_1);
            float2 _f32x2_mul_b_41 = make_float2(x0_1, x1_1);
            float2 gx_0;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_41));
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
            float2 _f32x2_mul_a_42 = make_float2(g0_2, g1_2);
            float2 _f32x2_mul_b_43 = make_float2(x0_2, x1_2);
            float2 gx_1;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_43));
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
            float2 _f32x2_mul_a_44 = make_float2(g0_3, g1_3);
            float2 _f32x2_mul_b_45 = make_float2(x0_3, x1_3);
            float2 gx_2;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
            float out_1[8];
            float g0_4 = __uint_as_float(gw[4] << 16);
            float g1_4 = __uint_as_float(gw[4] & 4294901760u);
            float x0_4 = __uint_as_float(xw[4] << 16);
            float x1_4 = __uint_as_float(xw[4] & 4294901760u);
            float2 _f32x2_mul_a_46 = make_float2(g0_4, g1_4);
            float2 _f32x2_mul_b_47 = make_float2(x0_4, x1_4);
            float2 gx_3;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_47));
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
            float2 _f32x2_mul_a_48 = make_float2(g0_5, g1_5);
            float2 _f32x2_mul_b_49 = make_float2(x0_5, x1_5);
            float2 gx_4;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_49));
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
            float2 _f32x2_mul_a_50 = make_float2(g0_6, g1_6);
            float2 _f32x2_mul_b_51 = make_float2(x0_6, x1_6);
            float2 gx_5;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
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
            float2 _f32x2_mul_a_52 = make_float2(g0_7, g1_7);
            float2 _f32x2_mul_b_53 = make_float2(x0_7, x1_7);
            float2 gx_6;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)((128 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
            float out_2[8];
            float g0_8 = __uint_as_float(gw[8] << 16);
            float g1_8 = __uint_as_float(gw[8] & 4294901760u);
            float x0_8 = __uint_as_float(xw[8] << 16);
            float x1_8 = __uint_as_float(xw[8] & 4294901760u);
            float2 _f32x2_mul_a_54 = make_float2(g0_8, g1_8);
            float2 _f32x2_mul_b_55 = make_float2(x0_8, x1_8);
            float2 gx_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
            float2 p_7;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_7) : "l"(*(const unsigned long long*)&gx_7), "l"(*(const unsigned long long*)&rr2));
            float2 a_7 = make_float2(acc[16], acc[17]);
            float2 s_7 = add_f32x2_noftz(a_7, p_7);
            float2 bn_7 = sub_f32x2_noftz(a_7, s_7);
            float2 av_7 = add_f32x2_noftz(s_7, bn_7);
            float2 db_7 = fma_f32x2_rn_noftz(gx_7, rr2, bn_7);
            float2 c_8 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[16], comp[17]), sub_f32x2_noftz(a_7, av_7)), db_7);
            acc[16] = s_7.x;
            acc[17] = s_7.y;
            comp[16] = c_8.x;
            comp[17] = c_8.y;
            float y0_8 = __fmaf_rn(nkx, x0_8, rq * (g0_8 * __uint_as_float(ww[8] << 16)));
            float y1_8 = __fmaf_rn(nkx, x1_8, rq * (g1_8 * __uint_as_float(ww[8] & 4294901760u)));
            out_2[0] = y0_8;
            out_2[1] = y1_8;
            float g0_9 = __uint_as_float(gw[9] << 16);
            float g1_9 = __uint_as_float(gw[9] & 4294901760u);
            float x0_9 = __uint_as_float(xw[9] << 16);
            float x1_9 = __uint_as_float(xw[9] & 4294901760u);
            float2 _f32x2_mul_a_56 = make_float2(g0_9, g1_9);
            float2 _f32x2_mul_b_57 = make_float2(x0_9, x1_9);
            float2 gx_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
            float2 p_8;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_8) : "l"(*(const unsigned long long*)&gx_8), "l"(*(const unsigned long long*)&rr2));
            float2 a_8 = make_float2(acc[18], acc[19]);
            float2 s_8 = add_f32x2_noftz(a_8, p_8);
            float2 bn_8 = sub_f32x2_noftz(a_8, s_8);
            float2 av_8 = add_f32x2_noftz(s_8, bn_8);
            float2 db_8 = fma_f32x2_rn_noftz(gx_8, rr2, bn_8);
            float2 c_9 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[18], comp[19]), sub_f32x2_noftz(a_8, av_8)), db_8);
            acc[18] = s_8.x;
            acc[19] = s_8.y;
            comp[18] = c_9.x;
            comp[19] = c_9.y;
            float y0_9 = __fmaf_rn(nkx, x0_9, rq * (g0_9 * __uint_as_float(ww[9] << 16)));
            float y1_9 = __fmaf_rn(nkx, x1_9, rq * (g1_9 * __uint_as_float(ww[9] & 4294901760u)));
            out_2[2] = y0_9;
            out_2[3] = y1_9;
            float g0_10 = __uint_as_float(gw[10] << 16);
            float g1_10 = __uint_as_float(gw[10] & 4294901760u);
            float x0_10 = __uint_as_float(xw[10] << 16);
            float x1_10 = __uint_as_float(xw[10] & 4294901760u);
            float2 _f32x2_mul_a_58 = make_float2(g0_10, g1_10);
            float2 _f32x2_mul_b_59 = make_float2(x0_10, x1_10);
            float2 gx_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
            float2 p_9;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_9) : "l"(*(const unsigned long long*)&gx_9), "l"(*(const unsigned long long*)&rr2));
            float2 a_9 = make_float2(acc[20], acc[21]);
            float2 s_9 = add_f32x2_noftz(a_9, p_9);
            float2 bn_9 = sub_f32x2_noftz(a_9, s_9);
            float2 av_9 = add_f32x2_noftz(s_9, bn_9);
            float2 db_9 = fma_f32x2_rn_noftz(gx_9, rr2, bn_9);
            float2 c_10 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[20], comp[21]), sub_f32x2_noftz(a_9, av_9)), db_9);
            acc[20] = s_9.x;
            acc[21] = s_9.y;
            comp[20] = c_10.x;
            comp[21] = c_10.y;
            float y0_10 = __fmaf_rn(nkx, x0_10, rq * (g0_10 * __uint_as_float(ww[10] << 16)));
            float y1_10 = __fmaf_rn(nkx, x1_10, rq * (g1_10 * __uint_as_float(ww[10] & 4294901760u)));
            out_2[4] = y0_10;
            out_2[5] = y1_10;
            float g0_11 = __uint_as_float(gw[11] << 16);
            float g1_11 = __uint_as_float(gw[11] & 4294901760u);
            float x0_11 = __uint_as_float(xw[11] << 16);
            float x1_11 = __uint_as_float(xw[11] & 4294901760u);
            float2 _f32x2_mul_a_60 = make_float2(g0_11, g1_11);
            float2 _f32x2_mul_b_61 = make_float2(x0_11, x1_11);
            float2 gx_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
            float2 p_10;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_10) : "l"(*(const unsigned long long*)&gx_10), "l"(*(const unsigned long long*)&rr2));
            float2 a_10 = make_float2(acc[22], acc[23]);
            float2 s_10 = add_f32x2_noftz(a_10, p_10);
            float2 bn_10 = sub_f32x2_noftz(a_10, s_10);
            float2 av_10 = add_f32x2_noftz(s_10, bn_10);
            float2 db_10 = fma_f32x2_rn_noftz(gx_10, rr2, bn_10);
            float2 c_11 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[22], comp[23]), sub_f32x2_noftz(a_10, av_10)), db_10);
            acc[22] = s_10.x;
            acc[23] = s_10.y;
            comp[22] = c_11.x;
            comp[23] = c_11.y;
            float y0_11 = __fmaf_rn(nkx, x0_11, rq * (g0_11 * __uint_as_float(ww[11] << 16)));
            float y1_11 = __fmaf_rn(nkx, x1_11, rq * (g1_11 * __uint_as_float(ww[11] & 4294901760u)));
            out_2[6] = y0_11;
            out_2[7] = y1_11;
            unsigned int packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_2[_lp*2 + 0], out_2[_lp*2+1 + 0]));
                packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)((256 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_2)[0];
            float out_3[8];
            float g0_12 = __uint_as_float(gw[12] << 16);
            float g1_12 = __uint_as_float(gw[12] & 4294901760u);
            float x0_12 = __uint_as_float(xw[12] << 16);
            float x1_12 = __uint_as_float(xw[12] & 4294901760u);
            float2 _f32x2_mul_a_62 = make_float2(g0_12, g1_12);
            float2 _f32x2_mul_b_63 = make_float2(x0_12, x1_12);
            float2 gx_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
            float2 p_11;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_11) : "l"(*(const unsigned long long*)&gx_11), "l"(*(const unsigned long long*)&rr2));
            float2 a_11 = make_float2(acc[24], acc[25]);
            float2 s_11 = add_f32x2_noftz(a_11, p_11);
            float2 bn_11 = sub_f32x2_noftz(a_11, s_11);
            float2 av_11 = add_f32x2_noftz(s_11, bn_11);
            float2 db_11 = fma_f32x2_rn_noftz(gx_11, rr2, bn_11);
            float2 c_12 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[24], comp[25]), sub_f32x2_noftz(a_11, av_11)), db_11);
            acc[24] = s_11.x;
            acc[25] = s_11.y;
            comp[24] = c_12.x;
            comp[25] = c_12.y;
            float y0_12 = __fmaf_rn(nkx, x0_12, rq * (g0_12 * __uint_as_float(ww[12] << 16)));
            float y1_12 = __fmaf_rn(nkx, x1_12, rq * (g1_12 * __uint_as_float(ww[12] & 4294901760u)));
            out_3[0] = y0_12;
            out_3[1] = y1_12;
            float g0_13 = __uint_as_float(gw[13] << 16);
            float g1_13 = __uint_as_float(gw[13] & 4294901760u);
            float x0_13 = __uint_as_float(xw[13] << 16);
            float x1_13 = __uint_as_float(xw[13] & 4294901760u);
            float2 _f32x2_mul_a_64 = make_float2(g0_13, g1_13);
            float2 _f32x2_mul_b_65 = make_float2(x0_13, x1_13);
            float2 gx_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
            float2 p_12;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_12) : "l"(*(const unsigned long long*)&gx_12), "l"(*(const unsigned long long*)&rr2));
            float2 a_12 = make_float2(acc[26], acc[27]);
            float2 s_12 = add_f32x2_noftz(a_12, p_12);
            float2 bn_12 = sub_f32x2_noftz(a_12, s_12);
            float2 av_12 = add_f32x2_noftz(s_12, bn_12);
            float2 db_12 = fma_f32x2_rn_noftz(gx_12, rr2, bn_12);
            float2 c_13 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[26], comp[27]), sub_f32x2_noftz(a_12, av_12)), db_12);
            acc[26] = s_12.x;
            acc[27] = s_12.y;
            comp[26] = c_13.x;
            comp[27] = c_13.y;
            float y0_13 = __fmaf_rn(nkx, x0_13, rq * (g0_13 * __uint_as_float(ww[13] << 16)));
            float y1_13 = __fmaf_rn(nkx, x1_13, rq * (g1_13 * __uint_as_float(ww[13] & 4294901760u)));
            out_3[2] = y0_13;
            out_3[3] = y1_13;
            float g0_14 = __uint_as_float(gw[14] << 16);
            float g1_14 = __uint_as_float(gw[14] & 4294901760u);
            float x0_14 = __uint_as_float(xw[14] << 16);
            float x1_14 = __uint_as_float(xw[14] & 4294901760u);
            float2 _f32x2_mul_a_66 = make_float2(g0_14, g1_14);
            float2 _f32x2_mul_b_67 = make_float2(x0_14, x1_14);
            float2 gx_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
            float2 p_13;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_13) : "l"(*(const unsigned long long*)&gx_13), "l"(*(const unsigned long long*)&rr2));
            float2 a_13 = make_float2(acc[28], acc[29]);
            float2 s_13 = add_f32x2_noftz(a_13, p_13);
            float2 bn_13 = sub_f32x2_noftz(a_13, s_13);
            float2 av_13 = add_f32x2_noftz(s_13, bn_13);
            float2 db_13 = fma_f32x2_rn_noftz(gx_13, rr2, bn_13);
            float2 c_14 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[28], comp[29]), sub_f32x2_noftz(a_13, av_13)), db_13);
            acc[28] = s_13.x;
            acc[29] = s_13.y;
            comp[28] = c_14.x;
            comp[29] = c_14.y;
            float y0_14 = __fmaf_rn(nkx, x0_14, rq * (g0_14 * __uint_as_float(ww[14] << 16)));
            float y1_14 = __fmaf_rn(nkx, x1_14, rq * (g1_14 * __uint_as_float(ww[14] & 4294901760u)));
            out_3[4] = y0_14;
            out_3[5] = y1_14;
            float g0_15 = __uint_as_float(gw[15] << 16);
            float g1_15 = __uint_as_float(gw[15] & 4294901760u);
            float x0_15 = __uint_as_float(xw[15] << 16);
            float x1_15 = __uint_as_float(xw[15] & 4294901760u);
            float2 _f32x2_mul_a_68 = make_float2(g0_15, g1_15);
            float2 _f32x2_mul_b_69 = make_float2(x0_15, x1_15);
            float2 gx_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&gx_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
            float2 p_14;
            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&p_14) : "l"(*(const unsigned long long*)&gx_14), "l"(*(const unsigned long long*)&rr2));
            float2 a_14 = make_float2(acc[30], acc[31]);
            float2 s_14 = add_f32x2_noftz(a_14, p_14);
            float2 bn_14 = sub_f32x2_noftz(a_14, s_14);
            float2 av_14 = add_f32x2_noftz(s_14, bn_14);
            float2 db_14 = fma_f32x2_rn_noftz(gx_14, rr2, bn_14);
            float2 c_15 = add_f32x2_noftz(add_f32x2_noftz(make_float2(comp[30], comp[31]), sub_f32x2_noftz(a_14, av_14)), db_14);
            acc[30] = s_14.x;
            acc[31] = s_14.y;
            comp[30] = c_15.x;
            comp[31] = c_15.y;
            float y0_15 = __fmaf_rn(nkx, x0_15, rq * (g0_15 * __uint_as_float(ww[15] << 16)));
            float y1_15 = __fmaf_rn(nkx, x1_15, rq * (g1_15 * __uint_as_float(ww[15] & 4294901760u)));
            out_3[6] = y0_15;
            out_3[7] = y1_15;
            unsigned int packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_3[_lp*2 + 0], out_3[_lp*2+1 + 0]));
                packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 4096ull + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_3)[0];
        }
        row0 = meta[2 * ((k_4 + 1) % 3)];
        nval_2 = meta[2 * ((k_4 + 1) % 3) + 1];
        if (nval_2 > rg) {
            rr[0] = r[row0 + rg];
        }
    }
    c = pc;
    pairs[2 * (rg * 4096 + tir)] = acc[0];
    pairs[2 * (rg * 4096 + tir) + 1] = comp[0];
    pairs[2 * (rg * 4096 + 128 + tir)] = acc[1];
    pairs[2 * (rg * 4096 + 128 + tir) + 1] = comp[1];
    pairs[2 * (rg * 4096 + 256 + tir)] = acc[2];
    pairs[2 * (rg * 4096 + 256 + tir) + 1] = comp[2];
    pairs[2 * (rg * 4096 + 384 + tir)] = acc[3];
    pairs[2 * (rg * 4096 + 384 + tir) + 1] = comp[3];
    pairs[2 * (rg * 4096 + 512 + tir)] = acc[4];
    pairs[2 * (rg * 4096 + 512 + tir) + 1] = comp[4];
    pairs[2 * (rg * 4096 + 640 + tir)] = acc[5];
    pairs[2 * (rg * 4096 + 640 + tir) + 1] = comp[5];
    pairs[2 * (rg * 4096 + 768 + tir)] = acc[6];
    pairs[2 * (rg * 4096 + 768 + tir) + 1] = comp[6];
    pairs[2 * (rg * 4096 + 896 + tir)] = acc[7];
    pairs[2 * (rg * 4096 + 896 + tir) + 1] = comp[7];
    pairs[2 * (rg * 4096 + 1024 + tir)] = acc[8];
    pairs[2 * (rg * 4096 + 1024 + tir) + 1] = comp[8];
    pairs[2 * (rg * 4096 + 1152 + tir)] = acc[9];
    pairs[2 * (rg * 4096 + 1152 + tir) + 1] = comp[9];
    pairs[2 * (rg * 4096 + 1280 + tir)] = acc[10];
    pairs[2 * (rg * 4096 + 1280 + tir) + 1] = comp[10];
    pairs[2 * (rg * 4096 + 1408 + tir)] = acc[11];
    pairs[2 * (rg * 4096 + 1408 + tir) + 1] = comp[11];
    pairs[2 * (rg * 4096 + 1536 + tir)] = acc[12];
    pairs[2 * (rg * 4096 + 1536 + tir) + 1] = comp[12];
    pairs[2 * (rg * 4096 + 1664 + tir)] = acc[13];
    pairs[2 * (rg * 4096 + 1664 + tir) + 1] = comp[13];
    pairs[2 * (rg * 4096 + 1792 + tir)] = acc[14];
    pairs[2 * (rg * 4096 + 1792 + tir) + 1] = comp[14];
    pairs[2 * (rg * 4096 + 1920 + tir)] = acc[15];
    pairs[2 * (rg * 4096 + 1920 + tir) + 1] = comp[15];
    pairs[2 * (rg * 4096 + 2048 + tir)] = acc[16];
    pairs[2 * (rg * 4096 + 2048 + tir) + 1] = comp[16];
    pairs[2 * (rg * 4096 + 2176 + tir)] = acc[17];
    pairs[2 * (rg * 4096 + 2176 + tir) + 1] = comp[17];
    pairs[2 * (rg * 4096 + 2304 + tir)] = acc[18];
    pairs[2 * (rg * 4096 + 2304 + tir) + 1] = comp[18];
    pairs[2 * (rg * 4096 + 2432 + tir)] = acc[19];
    pairs[2 * (rg * 4096 + 2432 + tir) + 1] = comp[19];
    pairs[2 * (rg * 4096 + 2560 + tir)] = acc[20];
    pairs[2 * (rg * 4096 + 2560 + tir) + 1] = comp[20];
    pairs[2 * (rg * 4096 + 2688 + tir)] = acc[21];
    pairs[2 * (rg * 4096 + 2688 + tir) + 1] = comp[21];
    pairs[2 * (rg * 4096 + 2816 + tir)] = acc[22];
    pairs[2 * (rg * 4096 + 2816 + tir) + 1] = comp[22];
    pairs[2 * (rg * 4096 + 2944 + tir)] = acc[23];
    pairs[2 * (rg * 4096 + 2944 + tir) + 1] = comp[23];
    pairs[2 * (rg * 4096 + 3072 + tir)] = acc[24];
    pairs[2 * (rg * 4096 + 3072 + tir) + 1] = comp[24];
    pairs[2 * (rg * 4096 + 3200 + tir)] = acc[25];
    pairs[2 * (rg * 4096 + 3200 + tir) + 1] = comp[25];
    pairs[2 * (rg * 4096 + 3328 + tir)] = acc[26];
    pairs[2 * (rg * 4096 + 3328 + tir) + 1] = comp[26];
    pairs[2 * (rg * 4096 + 3456 + tir)] = acc[27];
    pairs[2 * (rg * 4096 + 3456 + tir) + 1] = comp[27];
    pairs[2 * (rg * 4096 + 3584 + tir)] = acc[28];
    pairs[2 * (rg * 4096 + 3584 + tir) + 1] = comp[28];
    pairs[2 * (rg * 4096 + 3712 + tir)] = acc[29];
    pairs[2 * (rg * 4096 + 3712 + tir) + 1] = comp[29];
    pairs[2 * (rg * 4096 + 3840 + tir)] = acc[30];
    pairs[2 * (rg * 4096 + 3840 + tir) + 1] = comp[30];
    pairs[2 * (rg * 4096 + 3968 + tir)] = acc[31];
    pairs[2 * (rg * 4096 + 3968 + tir) + 1] = comp[31];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpbulk_dst_70 = reinterpret_cast<void*>(partial + (long long)(slot * 8192));
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                    :: "l"(_cpbulk_dst_70), "r"(pairs_addr), "r"((uint32_t)(32768))
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (1))), "r"(static_cast<unsigned int>((unsigned int)rows_mine)) : "memory");
        }
        int hs = -1;
        if (c - n_chunks < 256) {
            hs = c - n_chunks;
        }
        flags[3] = hs;
        if (c == n_chunks + (int)gridDim.x - 1) {
            *(reinterpret_cast<unsigned int*>(counters) + (0)) = 0;
        }
    }
    __syncthreads();
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
        __syncthreads();
        asm volatile("fence.acquire.gpu;" ::: "memory");
        unsigned int nw[1];
        {
            uint32_t _scalar_bits_71;
            asm volatile("ld.global.cg.b32 %0, [%1];"
                : "=r"(_scalar_bits_71) : "l"((const void*)(counters + (3))) : "memory");
            nw[0] = (unsigned int)_scalar_bits_71;
        }
        int grid = (int)num_bids;
        int helpers = ((grid) < (256) ? (grid) : (256));
        int cl = tid % 8;
        int rl = tid / 8;
        #pragma unroll 1
        for (int kk = 0; kk < 256; kk++) {
            int sl_5 = hs_c + kk * helpers;
            if (sl_5 >= 256) {
                break;
            }
            int at0 = sl_5 * 16 + 2 * cl;
            float p_15[4];
            p_15[0] = 0.0f;
            p_15[1] = 0.0f;
            p_15[2] = 0.0f;
            p_15[3] = 0.0f;
            float buf[32];
            #pragma unroll 1
            for (int s0 = rl; s0 < grid; s0 += 128) {
                if (s0 < grid) {
                    {
                        unsigned _v4_72_0;
                        unsigned _v4_72_1;
                        unsigned _v4_72_2;
                        unsigned _v4_72_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_72_0), "=r"(_v4_72_1), "=r"(_v4_72_2), "=r"(_v4_72_3) : "l"((const void*)(partial + (s0 * 8192 + 2 * at0))) : "memory");
                        buf[0 + 0] = __uint_as_float(_v4_72_0);
                        buf[0 + 1] = __uint_as_float(_v4_72_1);
                        buf[0 + 2] = __uint_as_float(_v4_72_2);
                        buf[0 + 3] = __uint_as_float(_v4_72_3);
                    }
                }
                if (s0 + 16 < grid) {
                    {
                        unsigned _v4_73_0;
                        unsigned _v4_73_1;
                        unsigned _v4_73_2;
                        unsigned _v4_73_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_73_0), "=r"(_v4_73_1), "=r"(_v4_73_2), "=r"(_v4_73_3) : "l"((const void*)(partial + ((s0 + 16) * 8192 + 2 * at0))) : "memory");
                        buf[4 + 0] = __uint_as_float(_v4_73_0);
                        buf[4 + 1] = __uint_as_float(_v4_73_1);
                        buf[4 + 2] = __uint_as_float(_v4_73_2);
                        buf[4 + 3] = __uint_as_float(_v4_73_3);
                    }
                }
                if (s0 + 32 < grid) {
                    {
                        unsigned _v4_74_0;
                        unsigned _v4_74_1;
                        unsigned _v4_74_2;
                        unsigned _v4_74_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_74_0), "=r"(_v4_74_1), "=r"(_v4_74_2), "=r"(_v4_74_3) : "l"((const void*)(partial + ((s0 + 32) * 8192 + 2 * at0))) : "memory");
                        buf[8 + 0] = __uint_as_float(_v4_74_0);
                        buf[8 + 1] = __uint_as_float(_v4_74_1);
                        buf[8 + 2] = __uint_as_float(_v4_74_2);
                        buf[8 + 3] = __uint_as_float(_v4_74_3);
                    }
                }
                if (s0 + 48 < grid) {
                    {
                        unsigned _v4_75_0;
                        unsigned _v4_75_1;
                        unsigned _v4_75_2;
                        unsigned _v4_75_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_75_0), "=r"(_v4_75_1), "=r"(_v4_75_2), "=r"(_v4_75_3) : "l"((const void*)(partial + ((s0 + 48) * 8192 + 2 * at0))) : "memory");
                        buf[12 + 0] = __uint_as_float(_v4_75_0);
                        buf[12 + 1] = __uint_as_float(_v4_75_1);
                        buf[12 + 2] = __uint_as_float(_v4_75_2);
                        buf[12 + 3] = __uint_as_float(_v4_75_3);
                    }
                }
                if (s0 + 64 < grid) {
                    {
                        unsigned _v4_76_0;
                        unsigned _v4_76_1;
                        unsigned _v4_76_2;
                        unsigned _v4_76_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_76_0), "=r"(_v4_76_1), "=r"(_v4_76_2), "=r"(_v4_76_3) : "l"((const void*)(partial + ((s0 + 64) * 8192 + 2 * at0))) : "memory");
                        buf[16 + 0] = __uint_as_float(_v4_76_0);
                        buf[16 + 1] = __uint_as_float(_v4_76_1);
                        buf[16 + 2] = __uint_as_float(_v4_76_2);
                        buf[16 + 3] = __uint_as_float(_v4_76_3);
                    }
                }
                if (s0 + 80 < grid) {
                    {
                        unsigned _v4_77_0;
                        unsigned _v4_77_1;
                        unsigned _v4_77_2;
                        unsigned _v4_77_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_77_0), "=r"(_v4_77_1), "=r"(_v4_77_2), "=r"(_v4_77_3) : "l"((const void*)(partial + ((s0 + 80) * 8192 + 2 * at0))) : "memory");
                        buf[20 + 0] = __uint_as_float(_v4_77_0);
                        buf[20 + 1] = __uint_as_float(_v4_77_1);
                        buf[20 + 2] = __uint_as_float(_v4_77_2);
                        buf[20 + 3] = __uint_as_float(_v4_77_3);
                    }
                }
                if (s0 + 96 < grid) {
                    {
                        unsigned _v4_78_0;
                        unsigned _v4_78_1;
                        unsigned _v4_78_2;
                        unsigned _v4_78_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_78_0), "=r"(_v4_78_1), "=r"(_v4_78_2), "=r"(_v4_78_3) : "l"((const void*)(partial + ((s0 + 96) * 8192 + 2 * at0))) : "memory");
                        buf[24 + 0] = __uint_as_float(_v4_78_0);
                        buf[24 + 1] = __uint_as_float(_v4_78_1);
                        buf[24 + 2] = __uint_as_float(_v4_78_2);
                        buf[24 + 3] = __uint_as_float(_v4_78_3);
                    }
                }
                if (s0 + 112 < grid) {
                    {
                        unsigned _v4_79_0;
                        unsigned _v4_79_1;
                        unsigned _v4_79_2;
                        unsigned _v4_79_3;
                        asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_v4_79_0), "=r"(_v4_79_1), "=r"(_v4_79_2), "=r"(_v4_79_3) : "l"((const void*)(partial + ((s0 + 112) * 8192 + 2 * at0))) : "memory");
                        buf[28 + 0] = __uint_as_float(_v4_79_0);
                        buf[28 + 1] = __uint_as_float(_v4_79_1);
                        buf[28 + 2] = __uint_as_float(_v4_79_2);
                        buf[28 + 3] = __uint_as_float(_v4_79_3);
                    }
                }
                if (s0 < (int)nw[0]) {
                    float s_15 = p_15[0] + buf[0];
                    float bb = s_15 - p_15[0];
                    float err = p_15[0] - (s_15 - bb) + (buf[0] - bb);
                    p_15[1] = p_15[1] + buf[1] + err;
                    p_15[0] = s_15;
                    float s_15_1 = p_15[2] + buf[2];
                    float bb_0 = s_15_1 - p_15[2];
                    float err_0 = p_15[2] - (s_15_1 - bb_0) + (buf[2] - bb_0);
                    p_15[3] = p_15[3] + buf[3] + err_0;
                    p_15[2] = s_15_1;
                }
                if (s0 + 16 < (int)nw[0]) {
                    float s_16 = p_15[0] + buf[4];
                    float bb_1 = s_16 - p_15[0];
                    float err_1 = p_15[0] - (s_16 - bb_1) + (buf[4] - bb_1);
                    p_15[1] = p_15[1] + buf[5] + err_1;
                    p_15[0] = s_16;
                    float s_16_1 = p_15[2] + buf[6];
                    float bb_1_1 = s_16_1 - p_15[2];
                    float err_1_1 = p_15[2] - (s_16_1 - bb_1_1) + (buf[6] - bb_1_1);
                    p_15[3] = p_15[3] + buf[7] + err_1_1;
                    p_15[2] = s_16_1;
                }
                if (s0 + 32 < (int)nw[0]) {
                    float s_17 = p_15[0] + buf[8];
                    float bb_2 = s_17 - p_15[0];
                    float err_2 = p_15[0] - (s_17 - bb_2) + (buf[8] - bb_2);
                    p_15[1] = p_15[1] + buf[9] + err_2;
                    p_15[0] = s_17;
                    float s_17_1 = p_15[2] + buf[10];
                    float bb_2_1 = s_17_1 - p_15[2];
                    float err_2_1 = p_15[2] - (s_17_1 - bb_2_1) + (buf[10] - bb_2_1);
                    p_15[3] = p_15[3] + buf[11] + err_2_1;
                    p_15[2] = s_17_1;
                }
                if (s0 + 48 < (int)nw[0]) {
                    float s_18 = p_15[0] + buf[12];
                    float bb_3 = s_18 - p_15[0];
                    float err_3 = p_15[0] - (s_18 - bb_3) + (buf[12] - bb_3);
                    p_15[1] = p_15[1] + buf[13] + err_3;
                    p_15[0] = s_18;
                    float s_18_1 = p_15[2] + buf[14];
                    float bb_3_1 = s_18_1 - p_15[2];
                    float err_3_1 = p_15[2] - (s_18_1 - bb_3_1) + (buf[14] - bb_3_1);
                    p_15[3] = p_15[3] + buf[15] + err_3_1;
                    p_15[2] = s_18_1;
                }
                if (s0 + 64 < (int)nw[0]) {
                    float s_19 = p_15[0] + buf[16];
                    float bb_4 = s_19 - p_15[0];
                    float err_4 = p_15[0] - (s_19 - bb_4) + (buf[16] - bb_4);
                    p_15[1] = p_15[1] + buf[17] + err_4;
                    p_15[0] = s_19;
                    float s_19_1 = p_15[2] + buf[18];
                    float bb_4_1 = s_19_1 - p_15[2];
                    float err_4_1 = p_15[2] - (s_19_1 - bb_4_1) + (buf[18] - bb_4_1);
                    p_15[3] = p_15[3] + buf[19] + err_4_1;
                    p_15[2] = s_19_1;
                }
                if (s0 + 80 < (int)nw[0]) {
                    float s_20 = p_15[0] + buf[20];
                    float bb_5 = s_20 - p_15[0];
                    float err_5 = p_15[0] - (s_20 - bb_5) + (buf[20] - bb_5);
                    p_15[1] = p_15[1] + buf[21] + err_5;
                    p_15[0] = s_20;
                    float s_20_1 = p_15[2] + buf[22];
                    float bb_5_1 = s_20_1 - p_15[2];
                    float err_5_1 = p_15[2] - (s_20_1 - bb_5_1) + (buf[22] - bb_5_1);
                    p_15[3] = p_15[3] + buf[23] + err_5_1;
                    p_15[2] = s_20_1;
                }
                if (s0 + 96 < (int)nw[0]) {
                    float s_21 = p_15[0] + buf[24];
                    float bb_6 = s_21 - p_15[0];
                    float err_6 = p_15[0] - (s_21 - bb_6) + (buf[24] - bb_6);
                    p_15[1] = p_15[1] + buf[25] + err_6;
                    p_15[0] = s_21;
                    float s_21_1 = p_15[2] + buf[26];
                    float bb_6_1 = s_21_1 - p_15[2];
                    float err_6_1 = p_15[2] - (s_21_1 - bb_6_1) + (buf[26] - bb_6_1);
                    p_15[3] = p_15[3] + buf[27] + err_6_1;
                    p_15[2] = s_21_1;
                }
                if (s0 + 112 < (int)nw[0]) {
                    float s_22 = p_15[0] + buf[28];
                    float bb_7 = s_22 - p_15[0];
                    float err_7 = p_15[0] - (s_22 - bb_7) + (buf[28] - bb_7);
                    p_15[1] = p_15[1] + buf[29] + err_7;
                    p_15[0] = s_22;
                    float s_22_1 = p_15[2] + buf[30];
                    float bb_7_1 = s_22_1 - p_15[2];
                    float err_7_1 = p_15[2] - (s_22_1 - bb_7_1) + (buf[30] - bb_7_1);
                    p_15[3] = p_15[3] + buf[31] + err_7_1;
                    p_15[2] = s_22_1;
                }
            }
            pairs[4 * tid] = p_15[0];
            pairs[4 * tid + 1] = p_15[1];
            pairs[4 * tid + 2] = p_15[2];
            pairs[4 * tid + 3] = p_15[3];
            __syncthreads();
            if (tid < 8) {
                float s_23 = p_15[0] + pairs[4 * (8 + cl)];
                float bb_8 = s_23 - p_15[0];
                float err_8 = p_15[0] - (s_23 - bb_8) + (pairs[4 * (8 + cl)] - bb_8);
                p_15[1] = p_15[1] + pairs[4 * (8 + cl) + 1] + err_8;
                p_15[0] = s_23;
                float s_23_1 = p_15[2] + pairs[4 * (8 + cl) + 2];
                float bb_8_1 = s_23_1 - p_15[2];
                float err_8_1 = p_15[2] - (s_23_1 - bb_8_1) + (pairs[4 * (8 + cl) + 2] - bb_8_1);
                p_15[3] = p_15[3] + pairs[4 * (8 + cl) + 3] + err_8_1;
                p_15[2] = s_23_1;
                float s_24 = p_15[0] + pairs[4 * (16 + cl)];
                float bb_9 = s_24 - p_15[0];
                float err_9 = p_15[0] - (s_24 - bb_9) + (pairs[4 * (16 + cl)] - bb_9);
                p_15[1] = p_15[1] + pairs[4 * (16 + cl) + 1] + err_9;
                p_15[0] = s_24;
                float s_25 = p_15[2] + pairs[4 * (16 + cl) + 2];
                float bb_10 = s_25 - p_15[2];
                float err_10 = p_15[2] - (s_25 - bb_10) + (pairs[4 * (16 + cl) + 2] - bb_10);
                p_15[3] = p_15[3] + pairs[4 * (16 + cl) + 3] + err_10;
                p_15[2] = s_25;
                float s_26 = p_15[0] + pairs[4 * (24 + cl)];
                float bb_11 = s_26 - p_15[0];
                float err_11 = p_15[0] - (s_26 - bb_11) + (pairs[4 * (24 + cl)] - bb_11);
                p_15[1] = p_15[1] + pairs[4 * (24 + cl) + 1] + err_11;
                p_15[0] = s_26;
                float s_27 = p_15[2] + pairs[4 * (24 + cl) + 2];
                float bb_12 = s_27 - p_15[2];
                float err_12 = p_15[2] - (s_27 - bb_12) + (pairs[4 * (24 + cl) + 2] - bb_12);
                p_15[3] = p_15[3] + pairs[4 * (24 + cl) + 3] + err_12;
                p_15[2] = s_27;
                float s_28 = p_15[0] + pairs[4 * (32 + cl)];
                float bb_13 = s_28 - p_15[0];
                float err_13 = p_15[0] - (s_28 - bb_13) + (pairs[4 * (32 + cl)] - bb_13);
                p_15[1] = p_15[1] + pairs[4 * (32 + cl) + 1] + err_13;
                p_15[0] = s_28;
                float s_29 = p_15[2] + pairs[4 * (32 + cl) + 2];
                float bb_14 = s_29 - p_15[2];
                float err_14 = p_15[2] - (s_29 - bb_14) + (pairs[4 * (32 + cl) + 2] - bb_14);
                p_15[3] = p_15[3] + pairs[4 * (32 + cl) + 3] + err_14;
                p_15[2] = s_29;
                float s_30 = p_15[0] + pairs[4 * (40 + cl)];
                float bb_15 = s_30 - p_15[0];
                float err_15 = p_15[0] - (s_30 - bb_15) + (pairs[4 * (40 + cl)] - bb_15);
                p_15[1] = p_15[1] + pairs[4 * (40 + cl) + 1] + err_15;
                p_15[0] = s_30;
                float s_31 = p_15[2] + pairs[4 * (40 + cl) + 2];
                float bb_16 = s_31 - p_15[2];
                float err_16 = p_15[2] - (s_31 - bb_16) + (pairs[4 * (40 + cl) + 2] - bb_16);
                p_15[3] = p_15[3] + pairs[4 * (40 + cl) + 3] + err_16;
                p_15[2] = s_31;
                float s_32 = p_15[0] + pairs[4 * (48 + cl)];
                float bb_17 = s_32 - p_15[0];
                float err_17 = p_15[0] - (s_32 - bb_17) + (pairs[4 * (48 + cl)] - bb_17);
                p_15[1] = p_15[1] + pairs[4 * (48 + cl) + 1] + err_17;
                p_15[0] = s_32;
                float s_33 = p_15[2] + pairs[4 * (48 + cl) + 2];
                float bb_18 = s_33 - p_15[2];
                float err_18 = p_15[2] - (s_33 - bb_18) + (pairs[4 * (48 + cl) + 2] - bb_18);
                p_15[3] = p_15[3] + pairs[4 * (48 + cl) + 3] + err_18;
                p_15[2] = s_33;
                float s_34 = p_15[0] + pairs[4 * (56 + cl)];
                float bb_19 = s_34 - p_15[0];
                float err_19 = p_15[0] - (s_34 - bb_19) + (pairs[4 * (56 + cl)] - bb_19);
                p_15[1] = p_15[1] + pairs[4 * (56 + cl) + 1] + err_19;
                p_15[0] = s_34;
                float s_35 = p_15[2] + pairs[4 * (56 + cl) + 2];
                float bb_20 = s_35 - p_15[2];
                float err_20 = p_15[2] - (s_35 - bb_20) + (pairs[4 * (56 + cl) + 2] - bb_20);
                p_15[3] = p_15[3] + pairs[4 * (56 + cl) + 3] + err_20;
                p_15[2] = s_35;
                float s_36 = p_15[0] + pairs[4 * (64 + cl)];
                float bb_21 = s_36 - p_15[0];
                float err_21 = p_15[0] - (s_36 - bb_21) + (pairs[4 * (64 + cl)] - bb_21);
                p_15[1] = p_15[1] + pairs[4 * (64 + cl) + 1] + err_21;
                p_15[0] = s_36;
                float s_37 = p_15[2] + pairs[4 * (64 + cl) + 2];
                float bb_22 = s_37 - p_15[2];
                float err_22 = p_15[2] - (s_37 - bb_22) + (pairs[4 * (64 + cl) + 2] - bb_22);
                p_15[3] = p_15[3] + pairs[4 * (64 + cl) + 3] + err_22;
                p_15[2] = s_37;
                float s_38 = p_15[0] + pairs[4 * (72 + cl)];
                float bb_23 = s_38 - p_15[0];
                float err_23 = p_15[0] - (s_38 - bb_23) + (pairs[4 * (72 + cl)] - bb_23);
                p_15[1] = p_15[1] + pairs[4 * (72 + cl) + 1] + err_23;
                p_15[0] = s_38;
                float s_39 = p_15[2] + pairs[4 * (72 + cl) + 2];
                float bb_24 = s_39 - p_15[2];
                float err_24 = p_15[2] - (s_39 - bb_24) + (pairs[4 * (72 + cl) + 2] - bb_24);
                p_15[3] = p_15[3] + pairs[4 * (72 + cl) + 3] + err_24;
                p_15[2] = s_39;
                float s_40 = p_15[0] + pairs[4 * (80 + cl)];
                float bb_25 = s_40 - p_15[0];
                float err_25 = p_15[0] - (s_40 - bb_25) + (pairs[4 * (80 + cl)] - bb_25);
                p_15[1] = p_15[1] + pairs[4 * (80 + cl) + 1] + err_25;
                p_15[0] = s_40;
                float s_41 = p_15[2] + pairs[4 * (80 + cl) + 2];
                float bb_26 = s_41 - p_15[2];
                float err_26 = p_15[2] - (s_41 - bb_26) + (pairs[4 * (80 + cl) + 2] - bb_26);
                p_15[3] = p_15[3] + pairs[4 * (80 + cl) + 3] + err_26;
                p_15[2] = s_41;
                float s_42 = p_15[0] + pairs[4 * (88 + cl)];
                float bb_27 = s_42 - p_15[0];
                float err_27 = p_15[0] - (s_42 - bb_27) + (pairs[4 * (88 + cl)] - bb_27);
                p_15[1] = p_15[1] + pairs[4 * (88 + cl) + 1] + err_27;
                p_15[0] = s_42;
                float s_43 = p_15[2] + pairs[4 * (88 + cl) + 2];
                float bb_28 = s_43 - p_15[2];
                float err_28 = p_15[2] - (s_43 - bb_28) + (pairs[4 * (88 + cl) + 2] - bb_28);
                p_15[3] = p_15[3] + pairs[4 * (88 + cl) + 3] + err_28;
                p_15[2] = s_43;
                float s_44 = p_15[0] + pairs[4 * (96 + cl)];
                float bb_29 = s_44 - p_15[0];
                float err_29 = p_15[0] - (s_44 - bb_29) + (pairs[4 * (96 + cl)] - bb_29);
                p_15[1] = p_15[1] + pairs[4 * (96 + cl) + 1] + err_29;
                p_15[0] = s_44;
                float s_45 = p_15[2] + pairs[4 * (96 + cl) + 2];
                float bb_30 = s_45 - p_15[2];
                float err_30 = p_15[2] - (s_45 - bb_30) + (pairs[4 * (96 + cl) + 2] - bb_30);
                p_15[3] = p_15[3] + pairs[4 * (96 + cl) + 3] + err_30;
                p_15[2] = s_45;
                float s_46 = p_15[0] + pairs[4 * (104 + cl)];
                float bb_31 = s_46 - p_15[0];
                float err_31 = p_15[0] - (s_46 - bb_31) + (pairs[4 * (104 + cl)] - bb_31);
                p_15[1] = p_15[1] + pairs[4 * (104 + cl) + 1] + err_31;
                p_15[0] = s_46;
                float s_47 = p_15[2] + pairs[4 * (104 + cl) + 2];
                float bb_32 = s_47 - p_15[2];
                float err_32 = p_15[2] - (s_47 - bb_32) + (pairs[4 * (104 + cl) + 2] - bb_32);
                p_15[3] = p_15[3] + pairs[4 * (104 + cl) + 3] + err_32;
                p_15[2] = s_47;
                float s_48 = p_15[0] + pairs[4 * (112 + cl)];
                float bb_33 = s_48 - p_15[0];
                float err_33 = p_15[0] - (s_48 - bb_33) + (pairs[4 * (112 + cl)] - bb_33);
                p_15[1] = p_15[1] + pairs[4 * (112 + cl) + 1] + err_33;
                p_15[0] = s_48;
                float s_49 = p_15[2] + pairs[4 * (112 + cl) + 2];
                float bb_34 = s_49 - p_15[2];
                float err_34 = p_15[2] - (s_49 - bb_34) + (pairs[4 * (112 + cl) + 2] - bb_34);
                p_15[3] = p_15[3] + pairs[4 * (112 + cl) + 3] + err_34;
                p_15[2] = s_49;
                float s_50 = p_15[0] + pairs[4 * (120 + cl)];
                float bb_35 = s_50 - p_15[0];
                float err_35 = p_15[0] - (s_50 - bb_35) + (pairs[4 * (120 + cl)] - bb_35);
                p_15[1] = p_15[1] + pairs[4 * (120 + cl) + 1] + err_35;
                p_15[0] = s_50;
                float s_51 = p_15[2] + pairs[4 * (120 + cl) + 2];
                float bb_36 = s_51 - p_15[2];
                float err_36 = p_15[2] - (s_51 - bb_36) + (pairs[4 * (120 + cl) + 2] - bb_36);
                p_15[3] = p_15[3] + pairs[4 * (120 + cl) + 3] + err_36;
                p_15[2] = s_51;
                *(reinterpret_cast<float*>(dw) + ((at0 / 128 / 8 * 128 + at0 % 128) * 8 + at0 / 128 % 8)) = p_15[0] + p_15[1];
                *(reinterpret_cast<float*>(dw) + (((at0 + 1) / 128 / 8 * 128 + (at0 + 1) % 128) * 8 + (at0 + 1) / 128 % 8)) = p_15[2] + p_15[3];
            }
            __syncthreads();
        }
        if (tid == 0) {
            int add2 = 1;
            if (hs_c == 0) {
                add2 = 257 - helpers;
            }
            unsigned int old2;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(old2) : "l"(&counters[2]), "r"(static_cast<uint32_t>(add2)) : "memory");
            if (old2 + (unsigned int)add2 == 256) {
                *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
                *(reinterpret_cast<unsigned int*>(counters) + (2)) = 0;
                *(reinterpret_cast<unsigned int*>(counters) + (3)) = 0;
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
