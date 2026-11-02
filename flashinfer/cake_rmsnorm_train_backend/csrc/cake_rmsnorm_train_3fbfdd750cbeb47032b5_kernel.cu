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
#define SMEM_FLAGS_OFF 1024
#define SMEM_FLAGS_STAGE_BYTES 32
#define SMEM_FLAGS_STRIDE 32
#define SMEM_DOTS_OFF 1056
#define SMEM_DOTS_STAGE_BYTES 96
#define SMEM_DOTS_STRIDE 96
#define SMEM_SUMS_OFF 1184
#define SMEM_SUMS_STAGE_BYTES 49152
#define SMEM_SUMS_STRIDE 49152
#define SMEM_DWBUF_OFF 1184
#define SMEM_DWBUF_STAGE_BYTES 24576
#define SMEM_DWBUF_STRIDE 24576
#define SMEM_META_OFF 1152
#define SMEM_META_STAGE_BYTES 32
#define SMEM_META_STRIDE 32
#define SMEM_RING_OFF 50432
#define SMEM_RING_STAGE_BYTES 147456
#define SMEM_RING_STRIDE 147456
#define SMEM_TOTAL 197888
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
kernel_cake_rmsnorm_train_3fbfdd750cbeb47032b5(__nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ w, float* __restrict__ r, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, double* __restrict__ partial, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ g_h, int T, int n_chunks, int rows_per_chunk, long long g_stride_p0, long long x_stride_p0, long long g_h_stride_p0)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* flags = reinterpret_cast<int*>(smem_raw + 1024);
    const int flags_addr = smem + 1024;
    float* dots = reinterpret_cast<float*>(smem_raw + 1056);
    const int dots_addr = smem + 1056;
    double* sums = reinterpret_cast<double*>(smem_raw + 1184);
    const int sums_addr = smem + 1184;
    float* dwbuf = reinterpret_cast<float*>(smem_raw + 1184);
    const int dwbuf_addr = smem + 1184;
    int* meta = reinterpret_cast<int*>(smem_raw + 1152);
    const int meta_addr = smem + 1152;
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 50432);
    const int ring_addr = smem + 50432;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
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
    int rg = warp / 12;
    int tir = tid - rg * 384;
    unsigned int ww[8];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(w + (unsigned long long)(tir * 8));
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&ww[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w + (unsigned long long)((384 + tir) * 8));
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&ww[4]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    float acc[16];
    float comp[1];
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
    int rows_mine = 0;
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
    int pc = 0;
    int prow = 0;
    int pend = 0;
    unsigned int pnxt = 0;
    int ring_at = (int)ring_addr;
    unsigned int gw[8];
    unsigned int xw[8];
    unsigned int hw[8];
    float rr[2];
    int row0_n = 0;
    int nval_n = 0;
    if (tid == 0) {
        pc = (int)nxt;
        first_c = pc;
        if (pc < n_chunks) {
            int stages = T / 1;
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
            int _min_3 = ((begin_it) < (T) ? (begin_it) : (T));
            prow = _min_3;
            int _min_4 = ((begin_it + span) < (T) ? (begin_it + span) : (T));
            pend = _min_4;
            unsigned int _atomic_fetch_add_5 = atomicAdd(&counters[0], 1);
            pnxt = _atomic_fetch_add_5;
        }
        int slot = 0;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_1 = T / 1;
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
                    int _min_6 = ((begin_it_1) < (T) ? (begin_it_1) : (T));
                    prow = _min_6;
                    int _min_7 = ((begin_it_1 + span_1) < (T) ? (begin_it_1 + span_1) : (T));
                    pend = _min_7;
                    unsigned int _atomic_fetch_add_8 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_8;
                }
            }
        }
        int nval = 0;
        if (prow < pend) {
            int _min_9 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval = _min_9;
        }
        meta[2 * slot] = prow;
        meta[2 * slot + 1] = nval;
        if (nval > 0) {
            mbarrier_arrive_expect_tx(full_addr, 36864);
            int sbase = ring_at + slot * 36864;
            int _min_10 = ((nval - 1) < (0) ? (nval - 1) : (0));
            long long row_i_0 = (long long)(prow + _min_10);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_0 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_0 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_0 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        prow = prow + nval;
        int slot_0 = 1;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_2 = T / 1;
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
                    int _min_11 = ((begin_it_2) < (T) ? (begin_it_2) : (T));
                    prow = _min_11;
                    int _min_12 = ((begin_it_2 + span_2) < (T) ? (begin_it_2 + span_2) : (T));
                    pend = _min_12;
                    unsigned int _atomic_fetch_add_13 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_13;
                }
            }
        }
        int nval_0 = 0;
        if (prow < pend) {
            int _min_14 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_0 = _min_14;
        }
        meta[2 * slot_0] = prow;
        meta[2 * slot_0 + 1] = nval_0;
        if (nval_0 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 8, 36864);
            int sbase_1 = ring_at + slot_0 * 36864;
            int _min_15 = ((nval_0 - 1) < (0) ? (nval_0 - 1) : (0));
            long long row_i_1 = (long long)(prow + _min_15);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_1 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_1 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_1 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_1 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + 8),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        prow = prow + nval_0;
        int slot_1 = 2;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_3 = T / 1;
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
                    int _min_16 = ((begin_it_3) < (T) ? (begin_it_3) : (T));
                    prow = _min_16;
                    int _min_17 = ((begin_it_3 + span_3) < (T) ? (begin_it_3 + span_3) : (T));
                    pend = _min_17;
                    unsigned int _atomic_fetch_add_18 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_18;
                }
            }
        }
        int nval_1 = 0;
        if (prow < pend) {
            int _min_19 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_1 = _min_19;
        }
        meta[2 * slot_1] = prow;
        meta[2 * slot_1 + 1] = nval_1;
        if (nval_1 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 16, 36864);
            int sbase_2 = ring_at + slot_1 * 36864;
            int _min_20 = ((nval_1 - 1) < (0) ? (nval_1 - 1) : (0));
            long long row_i_2 = (long long)(prow + _min_20);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_2 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_2 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_2 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_2 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + 16),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        prow = prow + nval_1;
        int slot_2 = 3;
        if (prow >= pend) {
            if (pc < n_chunks) {
                pc = (int)pnxt;
                if (pc < n_chunks) {
                    int stages_4 = T / 1;
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
                    for (int k_4 = 0; k_4 < 5; k_4++) {
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
                    int _min_21 = ((begin_it_4) < (T) ? (begin_it_4) : (T));
                    prow = _min_21;
                    int _min_22 = ((begin_it_4 + span_4) < (T) ? (begin_it_4 + span_4) : (T));
                    pend = _min_22;
                    unsigned int _atomic_fetch_add_23 = atomicAdd(&counters[0], 1);
                    pnxt = _atomic_fetch_add_23;
                }
            }
        }
        int nval_2 = 0;
        if (prow < pend) {
            int _min_24 = ((pend - prow) < (1) ? (pend - prow) : (1));
            nval_2 = _min_24;
        }
        meta[2 * slot_2] = prow;
        meta[2 * slot_2 + 1] = nval_2;
        if (nval_2 > 0) {
            mbarrier_arrive_expect_tx(full_addr + 24, 36864);
            int sbase_3 = ring_at + slot_2 * 36864;
            int _min_25 = ((nval_2 - 1) < (0) ? (nval_2 - 1) : (0));
            long long row_i_3 = (long long)(prow + _min_25);
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_3 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + 24),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_3 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + 24),
                   "l"(0x12F0000000000000ULL) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                " [%0], [%1], %2, [%3], %4;"
                :: "r"(sbase_3 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_3 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + 24),
                   "l"(0x12F0000000000000ULL) : "memory");
        }
        prow = prow + nval_2;
    }
    __syncthreads();
    int row0 = meta[0];
    int nval_3 = meta[1];
    rr[0] = 0.0f;
    if (nval_3 > rg) {
        rr[0] = r[row0 + rg];
    }
    float dot[1];
    #pragma unroll 1
    for (int k_5 = 0; k_5 < T + n_chunks; k_5++) {
        if (nval_3 == 0) {
            break;
        }
        rows_mine = rows_mine + nval_3;
        row0_n = meta[2 * ((k_5 + 1) % 4)];
        nval_n = meta[2 * ((k_5 + 1) % 4) + 1];
        if (nval_n > rg) {
            rr[1] = r[row0_n + rg];
        }
        int _stage_full = (int)((unsigned int)k_5 % 4);
        int _parity_full = (int)((unsigned int)k_5 / 4 % 2);
        mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
        int rbase = ring_at + k_5 % 4 * 36864 + rg * 12288;
        __int128_t _smem_raw_26;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_26) : "r"(rbase + tir * 8 * 2));
        gw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[0];
        gw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[1];
        gw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[2];
        gw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_26)[3];
        __int128_t _smem_raw_27;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_27) : "r"(rbase + 12288 + tir * 8 * 2));
        xw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[0];
        xw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[1];
        xw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[2];
        xw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_27)[3];
        __int128_t _smem_raw_28;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_28) : "r"(rbase + 24576 + tir * 8 * 2));
        hw[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[0];
        hw[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[1];
        hw[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[2];
        hw[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_28)[3];
        __int128_t _smem_raw_29;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_29) : "r"(rbase + (384 + tir) * 8 * 2));
        gw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[0];
        gw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[1];
        gw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[2];
        gw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[3];
        __int128_t _smem_raw_30;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_30) : "r"(rbase + 12288 + (384 + tir) * 8 * 2));
        xw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[0];
        xw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[1];
        xw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[2];
        xw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[3];
        __int128_t _smem_raw_31;
        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_31) : "r"(rbase + 24576 + (384 + tir) * 8 * 2));
        hw[4] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[0];
        hw[5] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[1];
        hw[6] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[2];
        hw[7] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[3];
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
        float _warp_reduce_32 = dq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_32 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_32, offset);
        dot[0] = _warp_reduce_32;
        if (lane == 0) {
            dots[(k_5 & 1) * 12 + warp] = dot[0];
        }
        __syncthreads();
        float tot = 0.0f;
        tot += dots[(k_5 & 1) * 12 + rg * 12];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 1];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 2];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 3];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 4];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 5];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 6];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 7];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 8];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 9];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 10];
        tot += dots[(k_5 & 1) * 12 + rg * 12 + 11];
        dot[0] = tot;
        if (tid == 0) {
            int slot_3 = (k_5 + 4) % 4;
            if (prow >= pend) {
                if (pc < n_chunks) {
                    pc = (int)pnxt;
                    if (pc < n_chunks) {
                        int stages_5 = T / 1;
                        int sl_5 = rows_per_chunk;
                        if (sl_5 > stages_5) {
                            sl_5 = stages_5;
                        }
                        if (sl_5 < 1) {
                            sl_5 = 1;
                        }
                        int share_5 = (stages_5 + sl_5 - 1) / sl_5;
                        int tail_5 = (share_5 + 11) / 12;
                        if (tail_5 < 1) {
                            tail_5 = 1;
                        }
                        int rem_5 = stages_5;
                        int size_5 = (share_5 + 1) / 2;
                        int first_5 = 0;
                        int base_5 = 0;
                        int span_5 = 0;
                        int halving_5 = 1;
                        int begin_it_5 = 0;
                        #pragma unroll 1
                        for (int k_0 = 0; k_0 < 5; k_0++) {
                            if (halving_5 != 0) {
                                if (size_5 > tail_5) {
                                    if (rem_5 >= sl_5 * size_5) {
                                        if (pc < first_5 + sl_5) {
                                            span_5 = size_5;
                                            begin_it_5 = base_5 + (pc - first_5) * size_5;
                                            halving_5 = 0;
                                        } else {
                                            first_5 = first_5 + sl_5;
                                            base_5 = base_5 + sl_5 * size_5;
                                            rem_5 = rem_5 - sl_5 * size_5;
                                            size_5 = (size_5 + 1) / 2;
                                        }
                                    } else {
                                        halving_5 = 0;
                                    }
                                } else {
                                    halving_5 = 0;
                                }
                            }
                        }
                        if (span_5 == 0) {
                            int quot_5 = rem_5 / sl_5;
                            int extra_5 = rem_5 - quot_5 * sl_5;
                            if (pc < first_5 + extra_5) {
                                span_5 = quot_5 + 1;
                                begin_it_5 = base_5 + (pc - first_5) * (quot_5 + 1);
                            } else if (pc < first_5 + sl_5) {
                                span_5 = quot_5;
                                begin_it_5 = base_5 + extra_5 * (quot_5 + 1) + (pc - first_5 - extra_5) * quot_5;
                            } else {
                                begin_it_5 = stages_5;
                            }
                        }
                        int _min_33 = ((begin_it_5) < (T) ? (begin_it_5) : (T));
                        prow = _min_33;
                        int _min_34 = ((begin_it_5 + span_5) < (T) ? (begin_it_5 + span_5) : (T));
                        pend = _min_34;
                        unsigned int _atomic_fetch_add_35 = atomicAdd(&counters[0], 1);
                        pnxt = _atomic_fetch_add_35;
                    }
                }
            }
            int nval_3_1 = 0;
            if (prow < pend) {
                int _min_36 = ((pend - prow) < (1) ? (pend - prow) : (1));
                nval_3_1 = _min_36;
            }
            meta[2 * slot_3] = prow;
            meta[2 * slot_3 + 1] = nval_3_1;
            if (nval_3_1 > 0) {
                mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8, 36864);
                int sbase_4 = ring_at + slot_3 * 36864;
                int _min_37 = ((nval_3_1 - 1) < (0) ? (nval_3_1 - 1) : (0));
                long long row_i_4 = (long long)(prow + _min_37);
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4), "l"(reinterpret_cast<const uint8_t*>(g + (row_i_4 * g_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 12288), "l"(reinterpret_cast<const uint8_t*>(x + (row_i_4 * x_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1], %2, [%3], %4;"
                    :: "r"(sbase_4 + 24576), "l"(reinterpret_cast<const uint8_t*>(g_h + (row_i_4 * gh_stride))), "r"((unsigned)(12288)), "r"(full_addr + ((int)((unsigned int)(k_5 + 4) % 4)) * 8),
                       "l"(0x12F0000000000000ULL) : "memory");
            }
            prow = prow + nval_3_1;
        }
        float rq = rr[0];
        float cq = __fdiv_rn(dot[0], 6144.0f);
        float kx = rq * rq * rq * cq;
        float nkx = -kx;
        if (row0 + rg < row0 + nval_3) {
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144 + (unsigned long long)(tir * 8)))[0] = reinterpret_cast<int4*>(packed_0)[0];
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
            reinterpret_cast<int4*>(dx + ((unsigned long long)(row0 + rg) * 6144 + (unsigned long long)((384 + tir) * 8)))[0] = reinterpret_cast<int4*>(packed_1)[0];
        }
        row0 = row0_n;
        nval_3 = nval_n;
        rr[0] = rr[1];
        if ((k_5 + 1 & 15) == 0) {
            sums[rg * 6144 + tir] = sums[rg * 6144 + tir] + (double)acc[0];
            acc[0] = 0.0f;
            sums[rg * 6144 + 384 + tir] = sums[rg * 6144 + 384 + tir] + (double)acc[1];
            acc[1] = 0.0f;
            sums[rg * 6144 + 768 + tir] = sums[rg * 6144 + 768 + tir] + (double)acc[2];
            acc[2] = 0.0f;
            sums[rg * 6144 + 1152 + tir] = sums[rg * 6144 + 1152 + tir] + (double)acc[3];
            acc[3] = 0.0f;
            sums[rg * 6144 + 1536 + tir] = sums[rg * 6144 + 1536 + tir] + (double)acc[4];
            acc[4] = 0.0f;
            sums[rg * 6144 + 1920 + tir] = sums[rg * 6144 + 1920 + tir] + (double)acc[5];
            acc[5] = 0.0f;
            sums[rg * 6144 + 2304 + tir] = sums[rg * 6144 + 2304 + tir] + (double)acc[6];
            acc[6] = 0.0f;
            sums[rg * 6144 + 2688 + tir] = sums[rg * 6144 + 2688 + tir] + (double)acc[7];
            acc[7] = 0.0f;
            sums[rg * 6144 + 3072 + tir] = sums[rg * 6144 + 3072 + tir] + (double)acc[8];
            acc[8] = 0.0f;
            sums[rg * 6144 + 3456 + tir] = sums[rg * 6144 + 3456 + tir] + (double)acc[9];
            acc[9] = 0.0f;
            sums[rg * 6144 + 3840 + tir] = sums[rg * 6144 + 3840 + tir] + (double)acc[10];
            acc[10] = 0.0f;
            sums[rg * 6144 + 4224 + tir] = sums[rg * 6144 + 4224 + tir] + (double)acc[11];
            acc[11] = 0.0f;
            sums[rg * 6144 + 4608 + tir] = sums[rg * 6144 + 4608 + tir] + (double)acc[12];
            acc[12] = 0.0f;
            sums[rg * 6144 + 4992 + tir] = sums[rg * 6144 + 4992 + tir] + (double)acc[13];
            acc[13] = 0.0f;
            sums[rg * 6144 + 5376 + tir] = sums[rg * 6144 + 5376 + tir] + (double)acc[14];
            acc[14] = 0.0f;
            sums[rg * 6144 + 5760 + tir] = sums[rg * 6144 + 5760 + tir] + (double)acc[15];
            acc[15] = 0.0f;
        }
    }
    sums[rg * 6144 + tir] = sums[rg * 6144 + tir] + (double)acc[0];
    acc[0] = 0.0f;
    sums[rg * 6144 + 384 + tir] = sums[rg * 6144 + 384 + tir] + (double)acc[1];
    acc[1] = 0.0f;
    sums[rg * 6144 + 768 + tir] = sums[rg * 6144 + 768 + tir] + (double)acc[2];
    acc[2] = 0.0f;
    sums[rg * 6144 + 1152 + tir] = sums[rg * 6144 + 1152 + tir] + (double)acc[3];
    acc[3] = 0.0f;
    sums[rg * 6144 + 1536 + tir] = sums[rg * 6144 + 1536 + tir] + (double)acc[4];
    acc[4] = 0.0f;
    sums[rg * 6144 + 1920 + tir] = sums[rg * 6144 + 1920 + tir] + (double)acc[5];
    acc[5] = 0.0f;
    sums[rg * 6144 + 2304 + tir] = sums[rg * 6144 + 2304 + tir] + (double)acc[6];
    acc[6] = 0.0f;
    sums[rg * 6144 + 2688 + tir] = sums[rg * 6144 + 2688 + tir] + (double)acc[7];
    acc[7] = 0.0f;
    sums[rg * 6144 + 3072 + tir] = sums[rg * 6144 + 3072 + tir] + (double)acc[8];
    acc[8] = 0.0f;
    sums[rg * 6144 + 3456 + tir] = sums[rg * 6144 + 3456 + tir] + (double)acc[9];
    acc[9] = 0.0f;
    sums[rg * 6144 + 3840 + tir] = sums[rg * 6144 + 3840 + tir] + (double)acc[10];
    acc[10] = 0.0f;
    sums[rg * 6144 + 4224 + tir] = sums[rg * 6144 + 4224 + tir] + (double)acc[11];
    acc[11] = 0.0f;
    sums[rg * 6144 + 4608 + tir] = sums[rg * 6144 + 4608 + tir] + (double)acc[12];
    acc[12] = 0.0f;
    sums[rg * 6144 + 4992 + tir] = sums[rg * 6144 + 4992 + tir] + (double)acc[13];
    acc[13] = 0.0f;
    sums[rg * 6144 + 5376 + tir] = sums[rg * 6144 + 5376 + tir] + (double)acc[14];
    acc[14] = 0.0f;
    sums[rg * 6144 + 5760 + tir] = sums[rg * 6144 + 5760 + tir] + (double)acc[15];
    acc[15] = 0.0f;
    c = pc;
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (first_c < n_chunks) {
            {
                void* _cpred_dst_38 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_38), "r"(sums_addr + 0), "r"((uint32_t)(12288))
                    : "memory");
            }
            {
                void* _cpred_dst_39 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 1536));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_39), "r"(sums_addr + 12288), "r"((uint32_t)(12288))
                    : "memory");
            }
            {
                void* _cpred_dst_40 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 3072));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_40), "r"(sums_addr + 24576), "r"((uint32_t)(12288))
                    : "memory");
            }
            {
                void* _cpred_dst_41 = reinterpret_cast<void*>(partial + (long long)((int)bid % 4 * 6144 + 4608));
                asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64"
                    " [%0], [%1], %2;"
                    :: "l"(_cpred_dst_41), "r"(sums_addr + 36864), "r"((uint32_t)(12288))
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(counters) + (1))), "r"(static_cast<unsigned int>((unsigned int)rows_mine)) : "memory");
        }
        int hs = -1;
        if (c - n_chunks < 16) {
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
        int helpers = (((int)num_bids) < (16) ? ((int)num_bids) : (16));
        #pragma unroll 1
        for (int kk = 0; kk < 16; kk++) {
            int sl_6 = hs_c + kk * helpers;
            if (sl_6 >= 16) {
                break;
            }
            int at_0 = sl_6 * 384 + tid;
            double tw_0[4];
            {
                uint64_t _scalar_bits_42;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_42) : "l"((const void*)(partial + (at_0))) : "memory");
                tw_0[0] = __longlong_as_double(_scalar_bits_42);
            }
            {
                uint64_t _scalar_bits_43;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_43) : "l"((const void*)(partial + (6144 + at_0))) : "memory");
                tw_0[1] = __longlong_as_double(_scalar_bits_43);
            }
            {
                uint64_t _scalar_bits_44;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_44) : "l"((const void*)(partial + (12288 + at_0))) : "memory");
                tw_0[2] = __longlong_as_double(_scalar_bits_44);
            }
            {
                uint64_t _scalar_bits_45;
                asm volatile("ld.global.cg.b64 %0, [%1];"
                    : "=l"(_scalar_bits_45) : "l"((const void*)(partial + (18432 + at_0))) : "memory");
                tw_0[3] = __longlong_as_double(_scalar_bits_45);
            }
            double tot2_0 = tw_0[0];
            tot2_0 = tot2_0 + tw_0[1];
            tot2_0 = tot2_0 + tw_0[2];
            tot2_0 = tot2_0 + tw_0[3];
            *(reinterpret_cast<float*>(dw) + ((at_0 / 384 / 8 * 384 + at_0 % 384) * 8 + at_0 / 384 % 8)) = (float)tot2_0;
            *reinterpret_cast<unsigned long long*>(partial + (at_0)) = (unsigned long long)((unsigned long long)(0.0));
            *reinterpret_cast<unsigned long long*>(partial + (6144 + at_0)) = (unsigned long long)((unsigned long long)(0.0));
            *reinterpret_cast<unsigned long long*>(partial + (12288 + at_0)) = (unsigned long long)((unsigned long long)(0.0));
            *reinterpret_cast<unsigned long long*>(partial + (18432 + at_0)) = (unsigned long long)((unsigned long long)(0.0));
        }
        __syncthreads();
        if (tid == 0) {
            int add2 = 1;
            if (hs_c == 0) {
                add2 = 17 - helpers;
            }
            unsigned int old2;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(old2) : "l"(&counters[2]), "r"(static_cast<uint32_t>(add2)) : "memory");
            if (old2 + (unsigned int)add2 == 16) {
                *(reinterpret_cast<unsigned int*>(counters) + (1)) = 0;
                *(reinterpret_cast<unsigned int*>(counters) + (2)) = 0;
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
