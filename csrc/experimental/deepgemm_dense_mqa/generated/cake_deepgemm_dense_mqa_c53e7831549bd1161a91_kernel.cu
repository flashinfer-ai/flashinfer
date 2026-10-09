/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt in this directory.

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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 3
#define NUM_KV_PIPE_STAGES 5
#define NUM_TMEM_PIPE_STAGES 3
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 4096
#define SMEM_SMEM_Q_STRIDE 4096
#define SMEM_SMEM_KV_OFF 13312
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_KV_SCALES_OFF 177152
#define SMEM_SMEM_KV_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_KV_SCALES_STRIDE 1024
#define SMEM_SMEM_WEIGHTS_OFF 182272
#define SMEM_SMEM_WEIGHTS_STAGE_BYTES 128
#define SMEM_SMEM_WEIGHTS_STRIDE 128
#define SMEM_SCHED_PREFIX_OFF 13312
#define SMEM_SCHED_PREFIX_STAGE_BYTES 16384
#define SMEM_SCHED_PREFIX_STRIDE 16384
#define SMEM_SCHED_WARP_SUMS_OFF 29696
#define SMEM_SCHED_WARP_SUMS_STAGE_BYTES 128
#define SMEM_SCHED_WARP_SUMS_STRIDE 128
#define SMEM_SCHED_BOUNDS_OFF 182656
#define SMEM_SCHED_BOUNDS_STAGE_BYTES 16
#define SMEM_SCHED_BOUNDS_STRIDE 16
#define SMEM_TOTAL 182784
#define K_NEXT_N 1
#define K_NEXT_N_ATOM 1
#define K_NUM_NEXT_N_ATOMS 1
#define K_CHUNK_SPLITS 1048576
#define K_NUM_HEADS 32
#define K_UMMA_N 32
#define K_Q_TILE_ROWS 32
#define K_Q_TX_BYTES 4224
#define K_BLOCK_KV 64
#define K_NUM_BLOCKS_PER_SPLIT 4
#define K_SCALE_TAIL_F32 2048
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_deepgemm_dense_mqa_c53e7831549bd1161a91(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KV_scales, const __grid_constant__ CUtensorMap Weights, float* __restrict__ Logits, int* __restrict__ context_lens, int* __restrict__ block_table, int block_table_stride, int batch_size, int num_sms, int stride_logits)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 24)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 88)
    #define umma_full_addr (mbar_base + 128)
    #define umma_empty_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_kv_addr = smem + 13312;
    float* smem_kv_scales = reinterpret_cast<float*>(smem_raw + 177152);
    const int smem_kv_scales_addr = smem + 177152;
    float* smem_weights = reinterpret_cast<float*>(smem_raw + 182272);
    const int smem_weights_addr = smem + 182272;
    int* sched_prefix = reinterpret_cast<int*>(smem_raw + 13312);
    const int sched_prefix_addr = smem + 13312;
    int* sched_warp_sums = reinterpret_cast<int*>(smem_raw + 29696);
    const int sched_warp_sums_addr = smem + 29696;
    int* sched_bounds = reinterpret_cast<int*>(smem_raw + 182656);
    const int sched_bounds_addr = smem + 182656;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Weights))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV_scales))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // q_empty: 3 barriers, init_count=288
            mbarrier_init(smem + 24, 288);
            mbarrier_init(smem + 32, 288);
            mbarrier_init(smem + 40, 288);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 5 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // kv_empty: 5 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // umma_empty: 3 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 10) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int sc_tid = tid;
    int sc_lane = lane;
    int sc_warp = warp;
    int sc_row_base = sc_tid * 11;
    int sc_local[11];
    int sc_running = 0;
    #pragma unroll
    for (int i = 0; i < 11; i++) {
        int sc_q = sc_row_base + i;
        int sc_nseg = 0;
        if (sc_q < batch_size) {
            int sc_ctx = context_lens[sc_q];
            sc_nseg = (sc_ctx + 256 - 1) / 256;
        }
        sc_running += sc_nseg;
        sc_local[i] = sc_running;
    }
    int sc_thread_total = sc_running;
    int sc_lane_sum = sc_thread_total;
    int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, sc_lane_sum, 1, 32);
    int sc_up = _shfl_up_0;
    if (sc_lane >= 1) {
        sc_lane_sum += sc_up;
    }
    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, sc_lane_sum, 2, 32);
    int sc_up_0 = _shfl_up_1;
    if (sc_lane >= 2) {
        sc_lane_sum += sc_up_0;
    }
    int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, sc_lane_sum, 4, 32);
    int sc_up_1 = _shfl_up_2;
    if (sc_lane >= 4) {
        sc_lane_sum += sc_up_1;
    }
    int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, sc_lane_sum, 8, 32);
    int sc_up_2 = _shfl_up_3;
    if (sc_lane >= 8) {
        sc_lane_sum += sc_up_2;
    }
    int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, sc_lane_sum, 16, 32);
    int sc_up_3 = _shfl_up_4;
    if (sc_lane >= 16) {
        sc_lane_sum += sc_up_3;
    }
    if (sc_lane == 31) {
        sched_warp_sums[sc_warp] = sc_lane_sum;
    }
    __syncthreads();
    int sc_warp_total = 0;
    if (sc_lane < 12) {
        sc_warp_total = sched_warp_sums[sc_lane];
    }
    int sc_warp_sum = sc_warp_total;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, sc_warp_sum, 1, 32);
    int sc_up2 = _shfl_up_5;
    if (sc_lane >= 1) {
        sc_warp_sum += sc_up2;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, sc_warp_sum, 2, 32);
    int sc_up2_4 = _shfl_up_6;
    if (sc_lane >= 2) {
        sc_warp_sum += sc_up2_4;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, sc_warp_sum, 4, 32);
    int sc_up2_5 = _shfl_up_7;
    if (sc_lane >= 4) {
        sc_warp_sum += sc_up2_5;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, sc_warp_sum, 8, 32);
    int sc_up2_6 = _shfl_up_8;
    if (sc_lane >= 8) {
        sc_warp_sum += sc_up2_6;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, sc_warp_sum, 16, 32);
    int sc_up2_7 = _shfl_up_9;
    if (sc_lane >= 16) {
        sc_warp_sum += sc_up2_7;
    }
    int _shfl_0 = __shfl_sync(0xFFFFFFFF, sc_warp_sum, 11);
    int sc_total = _shfl_0;
    int _shfl_1 = __shfl_sync(0xFFFFFFFF, sc_warp_sum - sc_warp_total, sc_warp);
    int sc_preceding = _shfl_1;
    int sc_offset = sc_lane_sum - sc_thread_total + sc_preceding;
    #pragma unroll
    for (int i_1 = 0; i_1 < 11; i_1++) {
        int sc_qo = sc_row_base + i_1;
        if (sc_qo < batch_size) {
            sched_prefix[sc_qo] = sc_local[i_1] + sc_offset;
        }
    }
    __syncthreads();
    if (sc_tid < 2) {
        int sc_sm = bid + sc_tid;
        int sc_qd = sc_total / num_sms;
        int sc_rd = sc_total % num_sms;
        int sc_min = ((sc_sm < sc_rd) ? sc_sm : sc_rd);
        int sc_start = sc_sm * sc_qd + sc_min;
        int sc_lo = 0;
        int sc_hi = batch_size;
        #pragma unroll
        for (int _ = 0; _ < 13; _++) {
            int sc_mid = (sc_lo + sc_hi) / 2;
            int sc_midr = ((sc_mid < batch_size) ? sc_mid : batch_size - 1);
            int sc_le = ((sc_start >= sched_prefix[sc_midr]) ? 1 : 0);
            int sc_inb = ((sc_mid < batch_size) ? 1 : 0);
            int sc_go = sc_le * sc_inb;
            sc_lo = ((sc_go != 0) ? sc_mid + 1 : sc_lo);
            sc_hi = ((sc_go == 0) ? sc_mid : sc_hi);
        }
        int sc_prev = ((sc_lo > 0) ? sched_prefix[sc_lo - 1] : 0);
        sched_bounds[sc_tid * 2] = sc_lo;
        sched_bounds[sc_tid * 2 + 1] = sc_start - sc_prev;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: math ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // math_main
            unsigned int wg_idx = make_warp_uniform(warp / 4);
            int math_thread_idx = wg_idx * 128 + (unsigned int)(warp % 4 * 32 + lane);
            unsigned int mt_q_stage = 0;
            unsigned int mt_kv_stage = 0;
            unsigned int mt_tmem_stage = wg_idx;
            int start_r = sched_bounds[0];
            int start_kv = sched_bounds[1];
            int end_r = sched_bounds[2];
            int end_kv = sched_bounds[3];
            int r_stop = ((end_kv > 0) ? end_r + 1 : end_r);
            unsigned int _phase_q_full = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (int req = start_r; req < r_stop; req++) {
                int ctx_last = context_lens[req * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits = (ctx_last + 256 - 1) / 256;
                int lo = ((req == start_r) ? start_kv : 0);
                int hi = ((req == end_r) ? end_kv : num_kv_splits);
                #pragma unroll 1
                for (int chunk_lo = lo; chunk_lo < hi; chunk_lo += K_CHUNK_SPLITS) {
                    int nxt = chunk_lo + K_CHUNK_SPLITS;
                    int ce = ((nxt < hi) ? nxt : hi);
                    #pragma unroll 1
                    for (int atom = 0; atom < K_NUM_NEXT_N_ATOMS; atom++) {
                        int tok_base = req * K_NEXT_N + atom * K_NEXT_N_ATOM;
                        mbarrier_wait(q_full_addr + (mt_q_stage) * 8, _phase_q_full);
                        float weights_reg[K_Q_TILE_ROWS];
                        int w_base = mt_q_stage * (unsigned int)K_Q_TILE_ROWS;
                        #pragma unroll
                        for (int wi = 0; wi < K_Q_TILE_ROWS; wi++) {
                            weights_reg[wi] = smem_weights[w_base + wi];
                        }
                        int ctx_qi[K_NEXT_N_ATOM];
                        #pragma unroll
                        for (int qp = 0; qp < K_NEXT_N_ATOM; qp++) {
                            ctx_qi[qp] = context_lens[tok_base + qp];
                        }
                        #pragma unroll 1
                        for (int s = chunk_lo; s < ce; s++) {
                            mbarrier_wait(kv_full_addr + (mt_kv_stage) * 8, _phase_kv_full);
                            int sc_base = mt_kv_stage * 256;
                            float scale_kv = smem_kv_scales[sc_base + math_thread_idx];
                            mbarrier_wait(umma_full_addr + (mt_tmem_stage) * 8, _phase_umma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(kv_empty_addr + (mt_kv_stage) * 8);
                            mt_kv_stage += 1;
                            if (mt_kv_stage == 5) { mt_kv_stage = 0; _phase_kv_full ^= 1; }
                            int kv_pos = s * 256 + math_thread_idx;
                            #pragma unroll
                            for (int qi = 0; qi < K_NEXT_N_ATOM; qi++) {
                                float _tmem_load_0[32];
                                tmem_ld_x16(&_tmem_load_0[0], taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                tmem_ld_x16(&_tmem_load_0[16], taddr + mt_tmem_stage * (unsigned int)K_UMMA_N + (unsigned int)(qi * K_NUM_HEADS) + (unsigned int)(warp % 4 * 32 << 16) + 16);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if (qi == K_NEXT_N_ATOM - 1) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    mbarrier_arrive(umma_empty_addr + (mt_tmem_stage) * 8);
                                }
                                float _relu_wsum_0;
                                {
                                    float2 _sum0 = make_float2(0.0f, 0.0f);
                                    float2 _sum1 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int _j = 0; _j < 32; _j += 4) {
                                        asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                            "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                            "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                            "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                            "mov.b64 _pb, {%3, %4};\n\t"
                                            "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                            : "+l"(*(unsigned long long*)&_sum0) : "f"(_tmem_load_0[0 + _j + 0]), "f"(_tmem_load_0[0 + _j + 1]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 0]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 1]));
                                        asm("{\n\t.reg .f32 _ab0, _ab1;\n\t.reg .b64 _pr, _pa, _pb;\n\t"
                                            "abs.f32 _ab0, %1;\n\tabs.f32 _ab1, %2;\n\t"
                                            "mov.b64 _pr, {%1, %2};\n\tmov.b64 _pa, {_ab0, _ab1};\n\t"
                                            "add.rn.f32x2 _pr, _pr, _pa;\n\t"
                                            "mov.b64 _pb, {%3, %4};\n\t"
                                            "fma.rn.f32x2 %0, _pr, _pb, %0;\n\t}"
                                            : "+l"(*(unsigned long long*)&_sum1) : "f"(_tmem_load_0[0 + _j + 2]), "f"(_tmem_load_0[0 + _j + 3]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 2]), "f"(weights_reg[qi * K_NUM_HEADS + _j + 3]));
                                    }
                                    float2 _sum;
                                    asm("add.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_sum) : "l"(*(const unsigned long long*)&_sum0), "l"(*(const unsigned long long*)&_sum1));
                                    _relu_wsum_0 = (_sum.x + _sum.y) * 0.5f;
                                }
                                float scaled = _relu_wsum_0 * scale_kv;
                                float result = ((kv_pos < ctx_qi[qi]) ? scaled : -CAKE_INF);
                                int out_elem = (tok_base + qi) * stride_logits + kv_pos;
                                *(reinterpret_cast<float*>(Logits + out_elem) + (0)) = result;
                            }
                            mt_tmem_stage += 1;
                            if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                            mt_tmem_stage += 1;
                            if (mt_tmem_stage == 3) { mt_tmem_stage = 0; _phase_umma_full ^= 1; }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(q_empty_addr + (mt_q_stage) * 8);
                        mt_q_stage += 1;
                        if (mt_q_stage == 3) { mt_q_stage = 0; _phase_q_full ^= 1; }
                    }
                }
            }
        }
    // ---- Role: load_q ----
    } else if (warp == 8) {
        { // load_q_main
            unsigned int lq_stage = 0;
            int start_r_1 = sched_bounds[0];
            int start_kv_1 = sched_bounds[1];
            int end_r_1 = sched_bounds[2];
            int end_kv_1 = sched_bounds[3];
            int r_stop_1 = ((end_kv_1 > 0) ? end_r_1 + 1 : end_r_1);
            unsigned int _phase_q_empty = 1;
            #pragma unroll 1
            for (int req_1 = start_r_1; req_1 < r_stop_1; req_1++) {
                int ctx_last_1 = context_lens[req_1 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_1 = (ctx_last_1 + 256 - 1) / 256;
                int lo_1 = ((req_1 == start_r_1) ? start_kv_1 : 0);
                int hi_1 = ((req_1 == end_r_1) ? end_kv_1 : num_kv_splits_1);
                #pragma unroll 1
                for (int chunk_lo_1 = lo_1; chunk_lo_1 < hi_1; chunk_lo_1 += K_CHUNK_SPLITS) {
                    #pragma unroll 1
                    for (int atom_1 = 0; atom_1 < K_NUM_NEXT_N_ATOMS; atom_1++) {
                        int tok_base_1 = req_1 * K_NEXT_N + atom_1 * K_NEXT_N_ATOM;
                        mbarrier_wait(q_empty_addr + (lq_stage) * 8, _phase_q_empty);
                        if (elect_sync()) {
                            tma_2d_gmem2smem(smem_q_addr + lq_stage * 4096, (&Q), 0, tok_base_1 * K_NUM_HEADS, q_full_addr + (lq_stage) * 8);
                            tma_2d_gmem2smem(smem_weights_addr + lq_stage * 128, (&Weights), 0, tok_base_1, q_full_addr + (lq_stage) * 8);
                            mbarrier_arrive_expect_tx(q_full_addr + (lq_stage) * 8, K_Q_TX_BYTES);
                        }
                        lq_stage += 1;
                        if (lq_stage == 3) { lq_stage = 0; _phase_q_empty ^= 1; }
                    }
                }
            }
        }
    // ---- Role: load_kv ----
    } else if (warp == 9) {
        { // load_kv_main
            unsigned int lk_stage = 0;
            int start_r_2 = sched_bounds[0];
            int start_kv_2 = sched_bounds[1];
            int end_r_2 = sched_bounds[2];
            int end_kv_2 = sched_bounds[3];
            int r_stop_2 = ((end_kv_2 > 0) ? end_r_2 + 1 : end_r_2);
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int req_2 = start_r_2; req_2 < r_stop_2; req_2++) {
                int ctx_last_2 = context_lens[req_2 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_2 = (ctx_last_2 + 256 - 1) / 256;
                int lo_2 = ((req_2 == start_r_2) ? start_kv_2 : 0);
                int hi_2 = ((req_2 == end_r_2) ? end_kv_2 : num_kv_splits_2);
                int bt_base = req_2 * block_table_stride;
                int num_kv_pages = (ctx_last_2 + K_BLOCK_KV - 1) / K_BLOCK_KV;
                int cached_blk = -1;
                int cached_coord = 0;
                #pragma unroll 1
                for (int chunk_lo_2 = lo_2; chunk_lo_2 < hi_2; chunk_lo_2 += K_CHUNK_SPLITS) {
                    int nxt_1 = chunk_lo_2 + K_CHUNK_SPLITS;
                    int ce_1 = ((nxt_1 < hi_2) ? nxt_1 : hi_2);
                    #pragma unroll 1
                    for (int _atom = 0; _atom < K_NUM_NEXT_N_ATOMS; _atom++) {
                        #pragma unroll 1
                        for (int s_1 = chunk_lo_2; s_1 < ce_1; s_1++) {
                            int page_base = s_1 * K_NUM_BLOCKS_PER_SPLIT;
                            int page_blk = page_base / 32;
                            if (page_blk != cached_blk) {
                                cached_blk = page_blk;
                                int page_off = page_blk * 32 + lane;
                                cached_coord = 0;
                                if (page_off < num_kv_pages) {
                                    cached_coord = block_table[bt_base + page_off];
                                }
                            }
                            int page_idx_reg[K_NUM_BLOCKS_PER_SPLIT];
                            int src_lane = page_base - page_blk * 32;
                            #pragma unroll
                            for (int i_2 = 0; i_2 < K_NUM_BLOCKS_PER_SPLIT; i_2++) {
                                int _shfl_2 = __shfl_sync(0xFFFFFFFF, cached_coord, src_lane + i_2);
                                page_idx_reg[i_2] = _shfl_2;
                            }
                            mbarrier_wait(kv_empty_addr + (lk_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                #pragma unroll
                                for (int i_3 = 0; i_3 < K_NUM_BLOCKS_PER_SPLIT; i_3++) {
                                    tma_3d_gmem2smem(smem_kv_addr + lk_stage * 32768 + (unsigned int)(i_3 * (K_BLOCK_KV * 128)), (&KV), 0, 0, page_idx_reg[i_3], kv_full_addr + (lk_stage) * 8);
                                    tma_2d_gmem2smem(smem_kv_scales_addr + lk_stage * 1024 + (unsigned int)(i_3 * (K_BLOCK_KV * 4)), (&KV_scales), K_SCALE_TAIL_F32, page_idx_reg[i_3], kv_full_addr + (lk_stage) * 8);
                                }
                                mbarrier_arrive_expect_tx(kv_full_addr + (lk_stage) * 8, 33792);
                            }
                            lk_stage += 1;
                            if (lk_stage == 5) { lk_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 10) {
        { // mma_main
            unsigned int mm_q_stage = 0;
            unsigned int mm_kv_stage = 0;
            unsigned int mm_tmem_stage = 0;
            int start_r_3 = sched_bounds[0];
            int start_kv_3 = sched_bounds[1];
            int end_r_3 = sched_bounds[2];
            int end_kv_3 = sched_bounds[3];
            int r_stop_3 = ((end_kv_3 > 0) ? end_r_3 + 1 : end_r_3);
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_kv_full_1 = 0;
            unsigned int _phase_umma_empty = 1;
            #pragma unroll 1
            for (int req_3 = start_r_3; req_3 < r_stop_3; req_3++) {
                int ctx_last_3 = context_lens[req_3 * K_NEXT_N + (K_NEXT_N - 1)];
                int num_kv_splits_3 = (ctx_last_3 + 256 - 1) / 256;
                int lo_3 = ((req_3 == start_r_3) ? start_kv_3 : 0);
                int hi_3 = ((req_3 == end_r_3) ? end_kv_3 : num_kv_splits_3);
                #pragma unroll 1
                for (int chunk_lo_3 = lo_3; chunk_lo_3 < hi_3; chunk_lo_3 += K_CHUNK_SPLITS) {
                    int nxt_2 = chunk_lo_3 + K_CHUNK_SPLITS;
                    int ce_2 = ((nxt_2 < hi_3) ? nxt_2 : hi_3);
                    #pragma unroll 1
                    for (int _atom_1 = 0; _atom_1 < K_NUM_NEXT_N_ATOMS; _atom_1++) {
                        mbarrier_wait(q_full_addr + (mm_q_stage) * 8, _phase_q_full_1);
                        #pragma unroll 1
                        for (int _s = chunk_lo_3; _s < ce_2; _s++) {
                            mbarrier_wait(kv_full_addr + (mm_kv_stage) * 8, _phase_kv_full_1);
                            if (elect_sync()) {
                                mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                                int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 256;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (mm_tmem_stage * 32))), "r"(0));
                                tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                                mm_tmem_stage += 1;
                                if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                                mbarrier_wait(umma_empty_addr + (mm_tmem_stage) * 8, _phase_umma_empty);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_1 = (((smem_kv_addr + 16384) >> 4) & 0x3FFF) + (mm_kv_stage) * 2048;
                                int _mma_b_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (mm_q_stage) * 256;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134742032;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_acc + (mm_tmem_stage * 32))), "r"(0));
                                tcgen05_commit(umma_full_addr + (mm_tmem_stage) * 8);
                                mm_tmem_stage += 1;
                                if (mm_tmem_stage == 3) { mm_tmem_stage = 0; _phase_umma_empty ^= 1; }
                            }
                            __syncwarp();
                            mm_kv_stage += 1;
                            if (mm_kv_stage == 5) { mm_kv_stage = 0; _phase_kv_full_1 ^= 1; }
                        }
                        mbarrier_arrive(q_empty_addr + (mm_q_stage) * 8);
                        mm_q_stage += 1;
                        if (mm_q_stage == 3) { mm_q_stage = 0; _phase_q_full_1 ^= 1; }
                    }
                }
            }
        }
    // ---- Role: idle ----
    } else if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 10) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
