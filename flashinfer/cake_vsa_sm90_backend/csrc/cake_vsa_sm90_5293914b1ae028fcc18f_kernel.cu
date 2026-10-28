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

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

#define CAKE_INF CUDART_INF_F
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define NUM_Q_PIPE_STAGES 2
#define NUM_META_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_K_SMEM_OFF 33792
#define SMEM_K_SMEM_STAGE_BYTES 32768
#define SMEM_K_SMEM_STRIDE 32768
#define SMEM_VT_SMEM_A_OFF 132096
#define SMEM_VT_SMEM_A_STAGE_BYTES 16384
#define SMEM_VT_SMEM_A_STRIDE 32768
#define SMEM_VT_SMEM_B_OFF 148480
#define SMEM_VT_SMEM_B_STAGE_BYTES 16384
#define SMEM_VT_SMEM_B_STRIDE 32768
#define SMEM_META_SMEM_OFF 230400
#define SMEM_META_SMEM_STAGE_BYTES 1152
#define SMEM_META_SMEM_STRIDE 1152
#define SMEM_TQ_SMEM_OFF 232064
#define SMEM_TQ_SMEM_STAGE_BYTES 16
#define SMEM_TQ_SMEM_STRIDE 16
#define SMEM_MERGE_O_OFF 17408
#define SMEM_MERGE_O_STAGE_BYTES 16384
#define SMEM_MERGE_O_STRIDE 16384
#define SMEM_MERGE_ML_OFF 231552
#define SMEM_MERGE_ML_STAGE_BYTES 512
#define SMEM_MERGE_ML_STRIDE 512
#define SMEM_SMEM_V8_OFF 148480
#define SMEM_SMEM_V8_STAGE_BYTES 16384
#define SMEM_SMEM_V8_STRIDE 16384
#define SMEM_SMEM_V9_OFF 181248
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_SMEM_V10_OFF 214016
#define SMEM_SMEM_V10_STAGE_BYTES 16384
#define SMEM_SMEM_V10_STRIDE 16384
#define SMEM_TOTAL 232192
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}







__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_vsa_sm90_5293914b1ae028fcc18f(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, int* __restrict__ meta, const CakeParamArray<int16_t, 1728> hdr, int tile_stride, int seqlen_q, int seqlen_k, float scale_log2, int* __restrict__ dbg, unsigned long long* __restrict__ tl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_ready_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 48)
    #define k_empty_addr (mbar_base + 72)
    #define v_empty_addr (mbar_base + 96)
    #define meta_full_addr (mbar_base + 120)
    #define meta_empty_addr (mbar_base + 136)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int k_smem_addr = smem + 33792;
    __nv_bfloat16* vt_smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int vt_smem_a_addr = smem + 132096;
    __nv_bfloat16* vt_smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int vt_smem_b_addr = smem + 148480;
    int* meta_smem = reinterpret_cast<int*>(smem_raw + 230400);
    const int meta_smem_addr = smem + 230400;
    int* tq_smem = reinterpret_cast<int*>(smem_raw + 232064);
    const int tq_smem_addr = smem + 232064;
    float* merge_o = reinterpret_cast<float*>(smem_raw + 17408);
    const int merge_o_addr = smem + 17408;
    float* merge_ml = reinterpret_cast<float*>(smem_raw + 231552);
    const int merge_ml_addr = smem + 231552;
    float* smem_v8 = reinterpret_cast<float*>(smem_raw + 148480);
    const int smem_v8_addr = smem + 148480;
    float* smem_v9 = reinterpret_cast<float*>(smem_raw + 181248);
    const int smem_v9_addr = smem + 181248;
    float* smem_v10 = reinterpret_cast<float*>(smem_raw + 214016);
    const int smem_v10_addr = smem + 214016;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Vt))) : "memory");
        }
    }
    __syncwarp();

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 1 barriers, init_count=8
            mbarrier_init(smem + 16, 8);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'k_pipe' ---
            // k_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // --- pipeline 'v_pipe' ---
            // v_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            // --- pipeline 'meta_pipe' ---
            // meta_full: 2 barriers, init_count=64
            mbarrier_init(smem + 120, 64);
            mbarrier_init(smem + 128, 64);
            // meta_empty: 2 barriers, init_count=10
            mbarrier_init(smem + 136, 10);
            mbarrier_init(smem + 144, 10);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
        { // producer_main
            int cta = bid;
            int row0 = cta * tile_stride;
            int hb = cta * 12;
            int g_ctas = gridDim.x;
            int q_bound = tile_stride + 256 * g_ctas + 1;
            if (warp == 1) {
                asm volatile("barrier.sync 1, 384;" ::: "memory");
            } else {
                asm volatile("barrier.arrive 1, 384;" ::: "memory");
            }
            if (warp >= 2) {
                int st_t = (warp - 2) * 32 + lane;
                if (warp == 2) {
                    if (elect_sync()) {
                        int h_head_q = hdr[hb];
                        int h_qb0 = hdr[hb + 1];
                        int h_qb1 = hdr[hb + 2];
                        int h_mode = hdr[hb + 3];
                        mbarrier_arrive_expect_tx(q_ready_addr, 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(h_head_q * seqlen_q + h_qb0 * 64), "r"(0),
                               "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                        if (h_mode != 1) {
                            mbarrier_arrive_expect_tx(q_ready_addr + 8, 16384);
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(q_smem_addr + 16384), "l"((&Q)), "r"(0), "r"((h_head_q + 1) * seqlen_q + h_qb0 * 64), "r"(0),
                                   "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                }
                int nt_st_ld = meta[row0 * 144 + 7];
                int slot = 0;
                int mrow = row0 * 144;
                int w0 = meta[mrow + st_t];
                int w1 = meta[mrow + st_t + 64];
                int w2 = 0;
                if (st_t + 128 < 144) {
                    w2 = meta[mrow + st_t + 128];
                }
                mbarrier_wait(meta_empty_addr, 1);
                if (warp == 2) {
                    int head = __shfl_sync(0xFFFFFFFF, w0, 0);
                    int qb0 = __shfl_sync(0xFFFFFFFF, w0, 1);
                    int qb1 = __shfl_sync(0xFFFFFFFF, w0, 2);
                    int mode_w = __shfl_sync(0xFFFFFFFF, w0, 3);
                    int valid_s = __shfl_sync(0xFFFFFFFF, w0, 7);
                    if (elect_sync()) {
                    }
                }
                meta_smem[slot * 144 + st_t] = w0;
                meta_smem[slot * 144 + st_t + 64] = w1;
                if (st_t + 128 < 144) {
                    meta_smem[slot * 144 + st_t + 128] = w2;
                }
                mbarrier_arrive(meta_full_addr);
                int nt_st = __shfl_sync(0xFFFFFFFF, nt_st_ld, 0);
                int pool_base = g_ctas * tile_stride;
                #pragma unroll 1
                for (int ti = 1; ti < q_bound; ti++) {
                    int trow = row0 + ti;
                    if (nt_st <= ti) {
                        if (warp == 2) {
                            if (elect_sync()) {
                                int tq_fetch = atomicAdd(&dbg[62], 1);
                                tq_smem[ti & 1] = tq_fetch + pool_base;
                            }
                        }
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                        trow = tq_smem[ti & 1];
                    }
                    int slot_0 = ti & 1;
                    int mrow_1 = trow * 144;
                    int w0_2 = meta[mrow_1 + st_t];
                    int w1_3 = meta[mrow_1 + st_t + 64];
                    int w2_4 = 0;
                    if (st_t + 128 < 144) {
                        w2_4 = meta[mrow_1 + st_t + 128];
                    }
                    int _stage_meta_empty = (int)((unsigned int)ti % 2);
                    int _parity_meta_empty = (int)((unsigned int)ti / 2 % 2) ^ 1;
                    mbarrier_wait(meta_empty_addr + (_stage_meta_empty) * 8, _parity_meta_empty);
                    if (warp == 2) {
                        int head_1 = __shfl_sync(0xFFFFFFFF, w0_2, 0);
                        int qb0_1 = __shfl_sync(0xFFFFFFFF, w0_2, 1);
                        int qb1_1 = __shfl_sync(0xFFFFFFFF, w0_2, 2);
                        int mode_w_1 = __shfl_sync(0xFFFFFFFF, w0_2, 3);
                        int valid_s_1 = __shfl_sync(0xFFFFFFFF, w0_2, 7);
                        if (elect_sync()) {
                            if (ti > 0) {
                                if (valid_s_1 != 0) {
                                    int _parity_q_empty = (int)((unsigned int)ti % 2) ^ 1;
                                    mbarrier_wait(q_empty_addr, _parity_q_empty);
                                    mbarrier_arrive_expect_tx(q_ready_addr, 16384);
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(head_1 * seqlen_q + qb0_1 * 64), "r"(0),
                                           "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                                    if (mode_w_1 != 1) {
                                        mbarrier_arrive_expect_tx(q_ready_addr + 8, 16384);
                                        asm volatile(
                                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                            :: "r"(q_smem_addr + 16384), "l"((&Q)), "r"(0), "r"((head_1 + 1) * seqlen_q + qb0_1 * 64), "r"(0),
                                               "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                                    }
                                }
                            }
                        }
                    }
                    meta_smem[slot_0 * 144 + st_t] = w0_2;
                    meta_smem[slot_0 * 144 + st_t + 64] = w1_3;
                    if (st_t + 128 < 144) {
                        meta_smem[slot_0 * 144 + st_t + 128] = w2_4;
                    }
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)ti % 2)) * 8);
                    int valid_st = meta[trow * 144 + 7];
                    if (valid_st == 0) {
                        if (warp == 2) {
                            if (elect_sync()) {
                                __threadfence();
                                int tq_done = atomicAdd(&dbg[63], 1);
                                if (tq_done == g_ctas - 1) {
                                    __threadfence();
                                    dbg[62] = 0;
                                    dbg[63] = 0;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    int gk = 0;
                    int kvh_k = hdr[hb + 11];
                    unsigned long long pol_k = ((kvh_k != 0) ? 1364590687093260288 : 1508705875169116160);
                    int h_head_k = hdr[hb + 2];
                    int n_hdr_k = hdr[hb + 4];
                    int kv_base_h = h_head_k * seqlen_k;
                    if (n_hdr_k > 0) {
                        mbarrier_wait(k_empty_addr, 1);
                        int hblk_a_1_0 = hdr[hb + 5];
                        int hblk_b_1_0 = hdr[hb + 6];
                        int kv_row_a = kv_base_h + hblk_a_1_0 * 64;
                        int kv_row_b = kv_base_h + hblk_b_1_0 * 64;
                        int has_b = 1;
                        if (hblk_b_1_0 < 0) {
                            has_b = 0;
                        }
                        int k_dst = k_smem_addr;
                        mbarrier_arrive_expect_tx(k_full_addr, 16384 * (1 + has_b));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst), "l"((&K)), "r"(0), "r"(kv_row_a), "r"(0),
                               "r"(k_full_addr), "l"(pol_k) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst + 16384), "l"((&K)), "r"(0), "r"(kv_row_a), "r"(1),
                               "r"(k_full_addr), "l"(pol_k) : "memory");
                        if (has_b != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst + 8192), "l"((&K)), "r"(0), "r"(kv_row_b), "r"(0),
                                   "r"(k_full_addr), "l"(pol_k) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst + 16384 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b), "r"(1),
                                   "r"(k_full_addr), "l"(pol_k) : "memory");
                        }
                    }
                    if (n_hdr_k > 1) {
                        mbarrier_wait(k_empty_addr + 8, 1);
                        int hblk_a_1_1 = hdr[hb + 5 + 2];
                        int hblk_b_1_1 = hdr[hb + 6 + 2];
                        int kv_row_a_1 = kv_base_h + hblk_a_1_1 * 64;
                        int kv_row_b_1 = kv_base_h + hblk_b_1_1 * 64;
                        int has_b_1 = 1;
                        if (hblk_b_1_1 < 0) {
                            has_b_1 = 0;
                        }
                        int k_dst_1 = k_smem_addr + 32768;
                        mbarrier_arrive_expect_tx(k_full_addr + 8, 16384 * (1 + has_b_1));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst_1), "l"((&K)), "r"(0), "r"(kv_row_a_1), "r"(0),
                               "r"(k_full_addr + 8), "l"(pol_k) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst_1 + 16384), "l"((&K)), "r"(0), "r"(kv_row_a_1), "r"(1),
                               "r"(k_full_addr + 8), "l"(pol_k) : "memory");
                        if (has_b_1 != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_1 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_1), "r"(0),
                                   "r"(k_full_addr + 8), "l"(pol_k) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_1 + 16384 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_1), "r"(1),
                                   "r"(k_full_addr + 8), "l"(pol_k) : "memory");
                        }
                    }
                    if (n_hdr_k > 2) {
                        mbarrier_wait(k_empty_addr + 16, 1);
                        int hblk_a_1_2 = hdr[hb + 5 + 4];
                        int hblk_b_1_2 = hdr[hb + 6 + 4];
                        int kv_row_a_2 = kv_base_h + hblk_a_1_2 * 64;
                        int kv_row_b_2 = kv_base_h + hblk_b_1_2 * 64;
                        int has_b_2 = 1;
                        if (hblk_b_1_2 < 0) {
                            has_b_2 = 0;
                        }
                        int k_dst_2 = k_smem_addr + 65536;
                        mbarrier_arrive_expect_tx(k_full_addr + 16, 16384 * (1 + has_b_2));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst_2), "l"((&K)), "r"(0), "r"(kv_row_a_2), "r"(0),
                               "r"(k_full_addr + 16), "l"(pol_k) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(k_dst_2 + 16384), "l"((&K)), "r"(0), "r"(kv_row_a_2), "r"(1),
                               "r"(k_full_addr + 16), "l"(pol_k) : "memory");
                        if (has_b_2 != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_2 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_2), "r"(0),
                                   "r"(k_full_addr + 16), "l"(pol_k) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_2 + 16384 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_2), "r"(1),
                                   "r"(k_full_addr + 16), "l"(pol_k) : "memory");
                        }
                    }
                    int nt_k = q_bound;
                    #pragma unroll 1
                    for (int ti_1 = 0; ti_1 < nt_k; ti_1++) {
                        int slot_k = ti_1 & 1;
                        int mb_k = slot_k * 144;
                        int _stage_meta_full = (int)((unsigned int)ti_1 % 2);
                        int _parity_meta_full = (int)((unsigned int)ti_1 / 2 % 2);
                        mbarrier_wait(meta_full_addr + (_stage_meta_full) * 8, _parity_meta_full);
                        int valid_k = meta_smem[mb_k + 7];
                        if (valid_k == 0) {
                            break;
                        }
                        int head_k = meta_smem[mb_k + 2];
                        int n_seq_k = meta_smem[mb_k + 4];
                        int kv_base = head_k * seqlen_k;
                        int i0_k = 0;
                        if (ti_1 == 0) {
                            i0_k = n_hdr_k;
                        }
                        #pragma unroll 1
                        for (int i = i0_k; i < n_seq_k; i++) {
                            int p = gk + i;
                            int stage = p % 3;
                            int plap = 0;
                            plap += p / 3;
                            int _parity_k_empty = (int)((unsigned int)plap % 2) ^ 1;
                            mbarrier_wait(k_empty_addr + (stage) * 8, _parity_k_empty);
                            int sw = meta_smem[mb_k + 8 + i];
                            int blk_a = sw << 16 >> 16;
                            int blk_b = sw >> 16;
                            int kv_row_a_3 = kv_base + blk_a * 64;
                            int kv_row_b_3 = kv_base + blk_b * 64;
                            int has_b_3 = 1;
                            if (blk_b < 0) {
                                has_b_3 = 0;
                            }
                            int k_dst_3 = k_smem_addr + (unsigned int)(stage * 32768);
                            mbarrier_arrive_expect_tx(k_full_addr + (stage) * 8, 16384 * (1 + has_b_3));
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_3), "l"((&K)), "r"(0), "r"(kv_row_a_3), "r"(0),
                                   "r"(k_full_addr + (stage) * 8), "l"(pol_k) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(k_dst_3 + 16384), "l"((&K)), "r"(0), "r"(kv_row_a_3), "r"(1),
                                   "r"(k_full_addr + (stage) * 8), "l"(pol_k) : "memory");
                            if (has_b_3 != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(k_dst_3 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_3), "r"(0),
                                       "r"(k_full_addr + (stage) * 8), "l"(pol_k) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(k_dst_3 + 16384 + 8192), "l"((&K)), "r"(0), "r"(kv_row_b_3), "r"(1),
                                       "r"(k_full_addr + (stage) * 8), "l"(pol_k) : "memory");
                            }
                        }
                        gk += n_seq_k;
                        mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_1 % 2)) * 8);
                    }
                }
            }
            if (warp == 1) {
                if (elect_sync()) {
                    int gv = 0;
                    int kvh_v = hdr[hb + 11];
                    unsigned long long pol_v = ((kvh_v != 0) ? 1364590687093260288 : 1508705875169116160);
                    int h_head_v = hdr[hb + 2];
                    int n_hdr_v = hdr[hb + 4];
                    int kv_base_hv = h_head_v * seqlen_k;
                    if (n_hdr_v > 0) {
                        mbarrier_wait(v_empty_addr, 1);
                        int hvblk_a_1_0 = hdr[hb + 5];
                        int hvblk_b_1_0 = hdr[hb + 6];
                        int v_row_a = kv_base_hv + hvblk_a_1_0 * 64;
                        int v_row_b = kv_base_hv + hvblk_b_1_0 * 64;
                        int v_has_b = 1;
                        if (hvblk_b_1_0 < 0) {
                            v_has_b = 0;
                        }
                        mbarrier_arrive_expect_tx(v_full_addr, 16384 * (1 + v_has_b));
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(vt_smem_a_addr), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_a / 8), "r"(0),
                               "r"(v_full_addr), "l"(pol_v) : "memory");
                        if (v_has_b != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(vt_smem_b_addr), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_b / 8), "r"(0),
                                   "r"(v_full_addr), "l"(pol_v) : "memory");
                        }
                    }
                    if (n_hdr_v > 1) {
                        mbarrier_wait(v_empty_addr + 8, 1);
                        int hvblk_a_1_1 = hdr[hb + 5 + 2];
                        int hvblk_b_1_1 = hdr[hb + 6 + 2];
                        int v_row_a_1 = kv_base_hv + hvblk_a_1_1 * 64;
                        int v_row_b_1 = kv_base_hv + hvblk_b_1_1 * 64;
                        int v_has_b_1 = 1;
                        if (hvblk_b_1_1 < 0) {
                            v_has_b_1 = 0;
                        }
                        mbarrier_arrive_expect_tx(v_full_addr + 8, 16384 * (1 + v_has_b_1));
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(vt_smem_a_addr + 32768), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_a_1 / 8), "r"(0),
                               "r"(v_full_addr + 8), "l"(pol_v) : "memory");
                        if (v_has_b_1 != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(vt_smem_b_addr + 32768), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_b_1 / 8), "r"(0),
                                   "r"(v_full_addr + 8), "l"(pol_v) : "memory");
                        }
                    }
                    if (n_hdr_v > 2) {
                        mbarrier_wait(v_empty_addr + 16, 1);
                        int hvblk_a_1_2 = hdr[hb + 5 + 4];
                        int hvblk_b_1_2 = hdr[hb + 6 + 4];
                        int v_row_a_2 = kv_base_hv + hvblk_a_1_2 * 64;
                        int v_row_b_2 = kv_base_hv + hvblk_b_1_2 * 64;
                        int v_has_b_2 = 1;
                        if (hvblk_b_1_2 < 0) {
                            v_has_b_2 = 0;
                        }
                        mbarrier_arrive_expect_tx(v_full_addr + 16, 16384 * (1 + v_has_b_2));
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(vt_smem_a_addr + 65536), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_a_2 / 8), "r"(0),
                               "r"(v_full_addr + 16), "l"(pol_v) : "memory");
                        if (v_has_b_2 != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(vt_smem_b_addr + 65536), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_b_2 / 8), "r"(0),
                                   "r"(v_full_addr + 16), "l"(pol_v) : "memory");
                        }
                    }
                    int nt_v = q_bound;
                    #pragma unroll 1
                    for (int ti_2 = 0; ti_2 < nt_v; ti_2++) {
                        int slot_v = ti_2 & 1;
                        int mb_v = slot_v * 144;
                        int _stage_meta_full_1 = (int)((unsigned int)ti_2 % 2);
                        int _parity_meta_full_1 = (int)((unsigned int)ti_2 / 2 % 2);
                        mbarrier_wait(meta_full_addr + (_stage_meta_full_1) * 8, _parity_meta_full_1);
                        int valid_v = meta_smem[mb_v + 7];
                        if (valid_v == 0) {
                            break;
                        }
                        int head_v = meta_smem[mb_v + 2];
                        int n_seq_v = meta_smem[mb_v + 4];
                        int kv_base_v = head_v * seqlen_k;
                        int i0_v = 0;
                        if (ti_2 == 0) {
                            i0_v = n_hdr_v;
                        }
                        #pragma unroll 1
                        for (int i_1 = i0_v; i_1 < n_seq_v; i_1++) {
                            int pv = gv + i_1;
                            int stage_v = pv % 3;
                            int pvlap = 0;
                            pvlap += pv / 3;
                            int _parity_v_empty = (int)((unsigned int)pvlap % 2) ^ 1;
                            mbarrier_wait(v_empty_addr + (stage_v) * 8, _parity_v_empty);
                            int svw = meta_smem[mb_v + 8 + i_1];
                            int vblk_a = svw << 16 >> 16;
                            int vblk_b = svw >> 16;
                            int v_row_a_3 = kv_base_v + vblk_a * 64;
                            int v_row_b_3 = kv_base_v + vblk_b * 64;
                            int v_has_b_3 = 1;
                            if (vblk_b < 0) {
                                v_has_b_3 = 0;
                            }
                            mbarrier_arrive_expect_tx(v_full_addr + (stage_v) * 8, 16384 * (1 + v_has_b_3));
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(vt_smem_a_addr + (unsigned int)(stage_v * 32768)), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_a_3 / 8), "r"(0),
                                   "r"(v_full_addr + (stage_v) * 8), "l"(pol_v) : "memory");
                            if (v_has_b_3 != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(vt_smem_b_addr + (unsigned int)(stage_v * 32768)), "l"((&Vt)), "r"(0), "r"(0), "r"(v_row_b_3 / 8), "r"(0),
                                       "r"(v_full_addr + (stage_v) * 8), "l"(pol_v) : "memory");
                            }
                        }
                        gv += n_seq_v;
                        mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_2 % 2)) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: main ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
        { // main_main
            int cta_m = bid;
            int warp_ld = warp - 4;
            int consumer_warp = __shfl_sync(0xFFFFFFFF, warp_ld, 0);
            int cwg = consumer_warp / 4;
            int warp_in_wg = consumer_warp % 4;
            int tid_wg = warp_in_wg * 32 + lane;
            int m0_local = warp_in_wg * 16 + lane / 4;
            int m1_local = m0_local + 8;
            int quad = warp_in_wg * 8 + lane / 4;
            int tid_c = cwg * 128 + tid_wg;
            smem_v8[tid_c] = 0.0f;
            smem_v8[256 + tid_c] = 0.0f;
            smem_v8[512 + tid_c] = 0.0f;
            smem_v8[768 + tid_c] = 0.0f;
            smem_v8[1024 + tid_c] = 0.0f;
            smem_v8[1280 + tid_c] = 0.0f;
            smem_v8[1536 + tid_c] = 0.0f;
            smem_v8[1792 + tid_c] = 0.0f;
            smem_v8[2048 + tid_c] = 0.0f;
            smem_v8[2304 + tid_c] = 0.0f;
            smem_v8[2560 + tid_c] = 0.0f;
            smem_v8[2816 + tid_c] = 0.0f;
            smem_v8[3072 + tid_c] = 0.0f;
            smem_v8[3328 + tid_c] = 0.0f;
            smem_v8[3584 + tid_c] = 0.0f;
            smem_v8[3840 + tid_c] = 0.0f;
            smem_v9[tid_c] = 0.0f;
            smem_v9[256 + tid_c] = 0.0f;
            smem_v9[512 + tid_c] = 0.0f;
            smem_v9[768 + tid_c] = 0.0f;
            smem_v9[1024 + tid_c] = 0.0f;
            smem_v9[1280 + tid_c] = 0.0f;
            smem_v9[1536 + tid_c] = 0.0f;
            smem_v9[1792 + tid_c] = 0.0f;
            smem_v9[2048 + tid_c] = 0.0f;
            smem_v9[2304 + tid_c] = 0.0f;
            smem_v9[2560 + tid_c] = 0.0f;
            smem_v9[2816 + tid_c] = 0.0f;
            smem_v9[3072 + tid_c] = 0.0f;
            smem_v9[3328 + tid_c] = 0.0f;
            smem_v9[3584 + tid_c] = 0.0f;
            smem_v9[3840 + tid_c] = 0.0f;
            smem_v10[tid_c] = 0.0f;
            smem_v10[256 + tid_c] = 0.0f;
            smem_v10[512 + tid_c] = 0.0f;
            smem_v10[768 + tid_c] = 0.0f;
            smem_v10[1024 + tid_c] = 0.0f;
            smem_v10[1280 + tid_c] = 0.0f;
            smem_v10[1536 + tid_c] = 0.0f;
            smem_v10[1792 + tid_c] = 0.0f;
            smem_v10[2048 + tid_c] = 0.0f;
            smem_v10[2304 + tid_c] = 0.0f;
            smem_v10[2560 + tid_c] = 0.0f;
            smem_v10[2816 + tid_c] = 0.0f;
            smem_v10[3072 + tid_c] = 0.0f;
            smem_v10[3328 + tid_c] = 0.0f;
            smem_v10[3584 + tid_c] = 0.0f;
            smem_v10[3840 + tid_c] = 0.0f;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.arrive 1, 384;" ::: "memory");
            float d_o[64];
            float d_qk[64];
            unsigned int p_bf16[32];
            float poly_tf[2];
            int poly_ti[2];
            unsigned int q_frag[32];
            float row_max0 = -CUDART_INF_F;
            float row_max1 = -CUDART_INF_F;
            float row_sum0 = 0.0f;
            float row_sum1 = 0.0f;
            int q_ld_row = warp_in_wg * 16 + lane % 8 + 8 * (lane / 8 % 2);
            int q_ld_col_lane = 8 * (lane / 16);
            int gbase = 0;
            int prev = -1;
            int g_ctas_m = gridDim.x;
            int nt_m = tile_stride + 256 * g_ctas_m + 1;
            int nx[11];
            nx[7] = 0;
            nx[8] = gbase;
            nx[9] = 0;
            nx[10] = 0;
            nx[10] = 1;
            int slot_n = 0;
            int mbase_n = slot_n * 144;
            nx[7] = 0;
            mbarrier_wait(meta_full_addr, 0);
            int valid_n = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 7], 0);
            if (valid_n != 0) {
                nx[7] = 1;
            }
            if (nx[7] != 0) {
                int _shfl_sync_0 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n], 0);
                nx[0] = _shfl_sync_0;
                int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 1], 0);
                nx[1] = _shfl_sync_1;
                int _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 2], 0);
                nx[2] = _shfl_sync_2;
                int _shfl_sync_3 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 3], 0);
                nx[3] = _shfl_sync_3;
                int _shfl_sync_4 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 4], 0);
                nx[4] = _shfl_sync_4;
                int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 5 + cwg], 0);
                nx[5] = _shfl_sync_5;
                int _shfl_sync_6 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n + 5 + 1 - cwg], 0);
                nx[6] = _shfl_sync_6;
            }
            if (nx[7] != 0) {
                int qs_n = ((nx[3] == 1) ? 0 : cwg);
                mbarrier_wait(q_ready_addr + (qs_n) * 8, 0);
                int q_half_base_1_0 = q_smem_addr + (unsigned int)(qs_n * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_0 = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                    : "r"((q_half_base_1_0 + (q_col_bytes_2_0 ^ (q_half_base_1_0 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_1 = q_smem_addr + (unsigned int)(qs_n * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_1 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[4]), "=r"(q_frag[5]), "=r"(q_frag[6]), "=r"(q_frag[7])
                    : "r"((q_half_base_1_1 + (q_col_bytes_2_1 ^ (q_half_base_1_1 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_2 = q_smem_addr + (unsigned int)(qs_n * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_2 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[8]), "=r"(q_frag[9]), "=r"(q_frag[10]), "=r"(q_frag[11])
                    : "r"((q_half_base_1_2 + (q_col_bytes_2_2 ^ (q_half_base_1_2 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_3 = q_smem_addr + (unsigned int)(qs_n * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_3 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[12]), "=r"(q_frag[13]), "=r"(q_frag[14]), "=r"(q_frag[15])
                    : "r"((q_half_base_1_3 + (q_col_bytes_2_3 ^ (q_half_base_1_3 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_4 = q_smem_addr + (unsigned int)(qs_n * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_4 = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[16]), "=r"(q_frag[17]), "=r"(q_frag[18]), "=r"(q_frag[19])
                    : "r"((q_half_base_1_4 + (q_col_bytes_2_4 ^ (q_half_base_1_4 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_5 = q_smem_addr + (unsigned int)(qs_n * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_5 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[20]), "=r"(q_frag[21]), "=r"(q_frag[22]), "=r"(q_frag[23])
                    : "r"((q_half_base_1_5 + (q_col_bytes_2_5 ^ (q_half_base_1_5 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_6 = q_smem_addr + (unsigned int)(qs_n * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_6 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[24]), "=r"(q_frag[25]), "=r"(q_frag[26]), "=r"(q_frag[27])
                    : "r"((q_half_base_1_6 + (q_col_bytes_2_6 ^ (q_half_base_1_6 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_1_7 = q_smem_addr + (unsigned int)(qs_n * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_2_7 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[28]), "=r"(q_frag[29]), "=r"(q_frag[30]), "=r"(q_frag[31])
                    : "r"((q_half_base_1_7 + (q_col_bytes_2_7 ^ (q_half_base_1_7 >> 7 & 7) << 4)))
                    : "memory");
                if (elect_sync()) {
                    mbarrier_arrive(q_empty_addr);
                }
                if (nx[5] > 0) {
                    int mbase_n_0 = 0;
                    int e0_n = meta_smem[mbase_n_0 + 76 + cwg * 34] & 65535;
                    int own_n = gbase + (e0_n >> 3);
                    nx[8] = own_n;
                    nx[9] = e0_n >> 2 & 1;
                    #pragma unroll 1
                    for (int p_1 = prev + 1; p_1 < nx[8]; p_1++) {
                        int plap_1 = 0;
                        plap_1 += p_1 / 3;
                        int _parity_k_full = (int)((unsigned int)plap_1 % 2);
                        mbarrier_wait(k_full_addr + (p_1 % 3) * 8, _parity_k_full);
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (p_1 % 3) * 8);
                        }
                        int _parity_v_full = (int)((unsigned int)plap_1 % 2);
                        mbarrier_wait(v_full_addr + (p_1 % 3) * 8, _parity_v_full);
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (p_1 % 3) * 8);
                        }
                    }
                    int kown = 0;
                    kown += own_n;
                    int klown = 0;
                    klown += own_n / 3;
                    int _parity_k_full_1 = (int)((unsigned int)klown % 2);
                    mbarrier_wait(k_full_addr + (kown % 3) * 8, _parity_k_full_1);
                }
            }
            int stage_n = nx[8] % 3;
            int real_n = 0;
            if (nx[7] != 0) {
                if (nx[5] > 0) {
                    real_n = 1;
                }
            }
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            uint64_t _wgmma_desc_7 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_n * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
            uint64_t _wgmma_desc_8 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_n * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_8 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_8);
            if (real_n != 0) {
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_0)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(_wgmma_b_0 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(_wgmma_b_0 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(_wgmma_b_0 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(_wgmma_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(_wgmma_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(_wgmma_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(_wgmma_b_1 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            } else {
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                    : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                    : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_0)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            }
            #pragma unroll 1
            for (int ti_3 = 0; ti_3 < nt_m; ti_3++) {
                int slot_m = ti_3 & 1;
                int mbase = slot_m * 144;
                if (nx[10] == 0) {
                    int slot_n_0 = ti_3 & 1;
                    int mbase_n_0_1 = slot_n_0 * 144;
                    nx[7] = 0;
                    int _stage_meta_full_2 = (int)((unsigned int)ti_3 % 2);
                    int _parity_meta_full_2 = (int)((unsigned int)ti_3 / 2 % 2);
                    mbarrier_wait(meta_full_addr + (_stage_meta_full_2) * 8, _parity_meta_full_2);
                    int valid_n_0 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 7], 0);
                    if (valid_n_0 != 0) {
                        nx[7] = 1;
                    }
                    if (nx[7] != 0) {
                        int _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1], 0);
                        nx[0] = _shfl_sync_9;
                        int _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 1], 0);
                        nx[1] = _shfl_sync_10;
                        int _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 2], 0);
                        nx[2] = _shfl_sync_11;
                        int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 3], 0);
                        nx[3] = _shfl_sync_12;
                        int _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 4], 0);
                        nx[4] = _shfl_sync_13;
                        int _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 5 + cwg], 0);
                        nx[5] = _shfl_sync_14;
                        int _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_0_1 + 5 + 1 - cwg], 0);
                        nx[6] = _shfl_sync_15;
                    }
                    if (nx[7] == 0) {
                        break;
                    }
                }
                int head_m = nx[0];
                int qb0_m = nx[1];
                int qb1_m = nx[2];
                int mode_m = nx[3];
                int n_seq_m = nx[4];
                int n_own = nx[5];
                int n_other = nx[6];
                int is_split = mode_m == 1;
                int my_qb = qb0_m;
                int own_base_w = mbase + 76 + cwg * 34;
                int pos0 = nx[8];
                int has2_0 = nx[9];
                d_o[0] = 0.0f;
                d_o[1] = 0.0f;
                d_o[2] = 0.0f;
                d_o[3] = 0.0f;
                d_o[4] = 0.0f;
                d_o[5] = 0.0f;
                d_o[6] = 0.0f;
                d_o[7] = 0.0f;
                d_o[8] = 0.0f;
                d_o[9] = 0.0f;
                d_o[10] = 0.0f;
                d_o[11] = 0.0f;
                d_o[12] = 0.0f;
                d_o[13] = 0.0f;
                d_o[14] = 0.0f;
                d_o[15] = 0.0f;
                d_o[16] = 0.0f;
                d_o[17] = 0.0f;
                d_o[18] = 0.0f;
                d_o[19] = 0.0f;
                d_o[20] = 0.0f;
                d_o[21] = 0.0f;
                d_o[22] = 0.0f;
                d_o[23] = 0.0f;
                d_o[24] = 0.0f;
                d_o[25] = 0.0f;
                d_o[26] = 0.0f;
                d_o[27] = 0.0f;
                d_o[28] = 0.0f;
                d_o[29] = 0.0f;
                d_o[30] = 0.0f;
                d_o[31] = 0.0f;
                d_o[32] = 0.0f;
                d_o[33] = 0.0f;
                d_o[34] = 0.0f;
                d_o[35] = 0.0f;
                d_o[36] = 0.0f;
                d_o[37] = 0.0f;
                d_o[38] = 0.0f;
                d_o[39] = 0.0f;
                d_o[40] = 0.0f;
                d_o[41] = 0.0f;
                d_o[42] = 0.0f;
                d_o[43] = 0.0f;
                d_o[44] = 0.0f;
                d_o[45] = 0.0f;
                d_o[46] = 0.0f;
                d_o[47] = 0.0f;
                d_o[48] = 0.0f;
                d_o[49] = 0.0f;
                d_o[50] = 0.0f;
                d_o[51] = 0.0f;
                d_o[52] = 0.0f;
                d_o[53] = 0.0f;
                d_o[54] = 0.0f;
                d_o[55] = 0.0f;
                d_o[56] = 0.0f;
                d_o[57] = 0.0f;
                d_o[58] = 0.0f;
                d_o[59] = 0.0f;
                d_o[60] = 0.0f;
                d_o[61] = 0.0f;
                d_o[62] = 0.0f;
                d_o[63] = 0.0f;
                row_max0 = -CUDART_INF_F;
                row_max1 = -CUDART_INF_F;
                row_sum0 = 0.0f;
                row_sum1 = 0.0f;
                if (nx[10] == 0) {
                    int qs_n_1 = ((nx[3] == 1) ? 0 : cwg);
                    int _parity_q_ready = (int)((unsigned int)ti_3 % 2);
                    mbarrier_wait(q_ready_addr + (qs_n_1) * 8, _parity_q_ready);
                    int q_half_base_1_8 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_8 = q_ld_col_lane * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                        : "r"((q_half_base_1_8 + (q_col_bytes_2_8 ^ (q_half_base_1_8 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_9 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_9 = (16 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[4]), "=r"(q_frag[5]), "=r"(q_frag[6]), "=r"(q_frag[7])
                        : "r"((q_half_base_1_9 + (q_col_bytes_2_9 ^ (q_half_base_1_9 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_10 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_10 = (32 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[8]), "=r"(q_frag[9]), "=r"(q_frag[10]), "=r"(q_frag[11])
                        : "r"((q_half_base_1_10 + (q_col_bytes_2_10 ^ (q_half_base_1_10 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_11 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_11 = (48 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[12]), "=r"(q_frag[13]), "=r"(q_frag[14]), "=r"(q_frag[15])
                        : "r"((q_half_base_1_11 + (q_col_bytes_2_11 ^ (q_half_base_1_11 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_12 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_12 = q_ld_col_lane * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[16]), "=r"(q_frag[17]), "=r"(q_frag[18]), "=r"(q_frag[19])
                        : "r"((q_half_base_1_12 + (q_col_bytes_2_12 ^ (q_half_base_1_12 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_13 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_13 = (16 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[20]), "=r"(q_frag[21]), "=r"(q_frag[22]), "=r"(q_frag[23])
                        : "r"((q_half_base_1_13 + (q_col_bytes_2_13 ^ (q_half_base_1_13 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_14 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_14 = (32 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[24]), "=r"(q_frag[25]), "=r"(q_frag[26]), "=r"(q_frag[27])
                        : "r"((q_half_base_1_14 + (q_col_bytes_2_14 ^ (q_half_base_1_14 >> 7 & 7) << 4)))
                        : "memory");
                    int q_half_base_1_15 = q_smem_addr + (unsigned int)(qs_n_1 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                    int q_col_bytes_2_15 = (48 + q_ld_col_lane) * 2;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(q_frag[28]), "=r"(q_frag[29]), "=r"(q_frag[30]), "=r"(q_frag[31])
                        : "r"((q_half_base_1_15 + (q_col_bytes_2_15 ^ (q_half_base_1_15 >> 7 & 7) << 4)))
                        : "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(q_empty_addr);
                    }
                }
                int cur_pos = pos0;
                int cur_has2 = has2_0;
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                if (n_own > 0) {
                    int kpos0 = 0;
                    int klap0 = 0;
                    if (nx[10] == 0) {
                        int mbase_n_0_2 = (ti_3 & 1) * 144;
                        int e0_n_1 = meta_smem[mbase_n_0_2 + 76 + cwg * 34] & 65535;
                        int own_n_1 = gbase + (e0_n_1 >> 3);
                        nx[8] = own_n_1;
                        nx[9] = e0_n_1 >> 2 & 1;
                        #pragma unroll 1
                        for (int p_2 = prev + 1; p_2 < nx[8]; p_2++) {
                            int plap_2 = 0;
                            plap_2 += p_2 / 3;
                            int _parity_k_full_2 = (int)((unsigned int)plap_2 % 2);
                            mbarrier_wait(k_full_addr + (p_2 % 3) * 8, _parity_k_full_2);
                            if (elect_sync()) {
                                mbarrier_arrive(k_empty_addr + (p_2 % 3) * 8);
                            }
                            int _parity_v_full_1 = (int)((unsigned int)plap_2 % 2);
                            mbarrier_wait(v_full_addr + (p_2 % 3) * 8, _parity_v_full_1);
                            if (elect_sync()) {
                                mbarrier_arrive(v_empty_addr + (p_2 % 3) * 8);
                            }
                        }
                        int kown_1 = 0;
                        kown_1 += own_n_1;
                        int klown_1 = 0;
                        klown_1 += own_n_1 / 3;
                        int _parity_k_full_3 = (int)((unsigned int)klown_1 % 2);
                        mbarrier_wait(k_full_addr + (kown_1 % 3) * 8, _parity_k_full_3);
                        pos0 = nx[8];
                        has2_0 = nx[9];
                        cur_pos = pos0;
                        cur_has2 = has2_0;
                        kpos0 += pos0;
                        klap0 += pos0 / 3;
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_16 = (((uint64_t)(((k_smem_addr + (unsigned int)(pos0 % 3 * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_16 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_16);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(wgmma_b)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(wgmma_b + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(wgmma_b + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(wgmma_b + 6)
                            : "memory");
                        uint64_t _wgmma_desc_17 = (((uint64_t)(((k_smem_addr + (unsigned int)(pos0 % 3 * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_17 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_17);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(wgmma_b_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(wgmma_b_1 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(wgmma_b_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(wgmma_b_1 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (kpos0 % 3) * 8);
                        }
                    } else {
                        kpos0 += pos0;
                        klap0 += pos0 / 3;
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (kpos0 % 3) * 8);
                        }
                    }
                    float f_max0 = -CUDART_INF_F;
                    float f_max1 = -CUDART_INF_F;
                    float m0 = -CUDART_INF_F;
                    float m1 = -CUDART_INF_F;
                    if (scale_log2 >= 0.0f) {
                        float max_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])), fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]))), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])), fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]))));
                        float max_1_0 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])), fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]))), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])), fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]))));
                        float max_1_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37])), fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]))), fmaxf(fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53])), fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]))));
                        float max_1_2 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39])), fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]))), fmaxf(fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55])), fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]))));
                        m0 = ((has2_0 != 0) ? fmaxf(max_1, max_1_1) : max_1);
                        m1 = ((has2_0 != 0) ? fmaxf(max_1_0, max_1_2) : max_1_0);
                        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        m0 = fmaxf(m0, _shfl_xor_sync_18);
                        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        m0 = fmaxf(m0, _shfl_xor_sync_19);
                        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        m1 = fmaxf(m1, _shfl_xor_sync_20);
                        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        m1 = fmaxf(m1, _shfl_xor_sync_21);
                    } else {
                        float min_1 = fminf(fminf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])), fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]))), fminf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])), fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]))));
                        float min_1_0 = fminf(fminf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])), fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]))), fminf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])), fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]))));
                        float min_1_1 = fminf(fminf(fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37])), fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]))), fminf(fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53])), fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]))));
                        float min_1_2 = fminf(fminf(fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39])), fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]))), fminf(fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55])), fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]))));
                        m0 = ((has2_0 != 0) ? fminf(min_1, min_1_1) : min_1);
                        m1 = ((has2_0 != 0) ? fminf(min_1_0, min_1_2) : min_1_0);
                        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        m0 = fminf(m0, _shfl_xor_sync_22);
                        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        m0 = fminf(m0, _shfl_xor_sync_23);
                        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        m1 = fminf(m1, _shfl_xor_sync_24);
                        float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        m1 = fminf(m1, _shfl_xor_sync_25);
                    }
                    f_max0 = m0;
                    f_max1 = m1;
                    row_max0 = f_max0 * scale_log2;
                    row_max1 = f_max1 * scale_log2;
                    float f_sum0 = 0.0f;
                    float f_sum1 = 0.0f;
                    float sa0 = 0.0f;
                    float sa1 = 0.0f;
                    float sb0 = 0.0f;
                    float sb1 = 0.0f;
                    float exp2 = approx_exp2(__fmaf_rn(d_qk[0], scale_log2, -row_max0));
                    float exp2_1 = approx_exp2(__fmaf_rn(d_qk[1], scale_log2, -row_max0));
                    d_qk[0] = exp2;
                    d_qk[1] = exp2_1;
                    sa0 += exp2 + exp2_1;
                    float exp2_2 = approx_exp2(__fmaf_rn(d_qk[2], scale_log2, -row_max1));
                    float exp2_3 = approx_exp2(__fmaf_rn(d_qk[3], scale_log2, -row_max1));
                    d_qk[2] = exp2_2;
                    d_qk[3] = exp2_3;
                    sa1 += exp2_2 + exp2_3;
                    float exp2_4 = approx_exp2(__fmaf_rn(d_qk[4], scale_log2, -row_max0));
                    float exp2_5 = approx_exp2(__fmaf_rn(d_qk[5], scale_log2, -row_max0));
                    d_qk[4] = exp2_4;
                    d_qk[5] = exp2_5;
                    sa0 += exp2_4 + exp2_5;
                    float exp2_6 = approx_exp2(__fmaf_rn(d_qk[6], scale_log2, -row_max1));
                    float exp2_7 = approx_exp2(__fmaf_rn(d_qk[7], scale_log2, -row_max1));
                    d_qk[6] = exp2_6;
                    d_qk[7] = exp2_7;
                    sa1 += exp2_6 + exp2_7;
                    float exp2_8 = approx_exp2(__fmaf_rn(d_qk[8], scale_log2, -row_max0));
                    float exp2_9 = approx_exp2(__fmaf_rn(d_qk[9], scale_log2, -row_max0));
                    d_qk[8] = exp2_8;
                    d_qk[9] = exp2_9;
                    sa0 += exp2_8 + exp2_9;
                    float exp2_10 = approx_exp2(__fmaf_rn(d_qk[10], scale_log2, -row_max1));
                    float exp2_11 = approx_exp2(__fmaf_rn(d_qk[11], scale_log2, -row_max1));
                    d_qk[10] = exp2_10;
                    d_qk[11] = exp2_11;
                    sa1 += exp2_10 + exp2_11;
                    float exp2_12 = approx_exp2(__fmaf_rn(d_qk[12], scale_log2, -row_max0));
                    float exp2_13 = approx_exp2(__fmaf_rn(d_qk[13], scale_log2, -row_max0));
                    d_qk[12] = exp2_12;
                    d_qk[13] = exp2_13;
                    sa0 += exp2_12 + exp2_13;
                    float exp2_14 = approx_exp2(__fmaf_rn(d_qk[14], scale_log2, -row_max1));
                    float exp2_15 = approx_exp2(__fmaf_rn(d_qk[15], scale_log2, -row_max1));
                    d_qk[14] = exp2_14;
                    d_qk[15] = exp2_15;
                    sa1 += exp2_14 + exp2_15;
                    float exp2_16 = approx_exp2(__fmaf_rn(d_qk[16], scale_log2, -row_max0));
                    float exp2_17 = approx_exp2(__fmaf_rn(d_qk[17], scale_log2, -row_max0));
                    d_qk[16] = exp2_16;
                    d_qk[17] = exp2_17;
                    sa0 += exp2_16 + exp2_17;
                    float exp2_18 = approx_exp2(__fmaf_rn(d_qk[18], scale_log2, -row_max1));
                    float exp2_19 = approx_exp2(__fmaf_rn(d_qk[19], scale_log2, -row_max1));
                    d_qk[18] = exp2_18;
                    d_qk[19] = exp2_19;
                    sa1 += exp2_18 + exp2_19;
                    float exp2_20 = approx_exp2(__fmaf_rn(d_qk[20], scale_log2, -row_max0));
                    float exp2_21 = approx_exp2(__fmaf_rn(d_qk[21], scale_log2, -row_max0));
                    d_qk[20] = exp2_20;
                    d_qk[21] = exp2_21;
                    sa0 += exp2_20 + exp2_21;
                    float exp2_22 = approx_exp2(__fmaf_rn(d_qk[22], scale_log2, -row_max1));
                    float exp2_23 = approx_exp2(__fmaf_rn(d_qk[23], scale_log2, -row_max1));
                    d_qk[22] = exp2_22;
                    d_qk[23] = exp2_23;
                    sa1 += exp2_22 + exp2_23;
                    float exp2_24 = approx_exp2(__fmaf_rn(d_qk[24], scale_log2, -row_max0));
                    float exp2_25 = approx_exp2(__fmaf_rn(d_qk[25], scale_log2, -row_max0));
                    d_qk[24] = exp2_24;
                    d_qk[25] = exp2_25;
                    sa0 += exp2_24 + exp2_25;
                    float exp2_26 = approx_exp2(__fmaf_rn(d_qk[26], scale_log2, -row_max1));
                    float exp2_27 = approx_exp2(__fmaf_rn(d_qk[27], scale_log2, -row_max1));
                    d_qk[26] = exp2_26;
                    d_qk[27] = exp2_27;
                    sa1 += exp2_26 + exp2_27;
                    float exp2_28 = approx_exp2(__fmaf_rn(d_qk[28], scale_log2, -row_max0));
                    float exp2_29 = approx_exp2(__fmaf_rn(d_qk[29], scale_log2, -row_max0));
                    d_qk[28] = exp2_28;
                    d_qk[29] = exp2_29;
                    sa0 += exp2_28 + exp2_29;
                    float exp2_30 = approx_exp2(__fmaf_rn(d_qk[30], scale_log2, -row_max1));
                    float exp2_31 = approx_exp2(__fmaf_rn(d_qk[31], scale_log2, -row_max1));
                    d_qk[30] = exp2_30;
                    d_qk[31] = exp2_31;
                    sa1 += exp2_30 + exp2_31;
                    float exp2_0 = approx_exp2(__fmaf_rn(d_qk[32], scale_log2, -row_max0));
                    float exp2_1_0 = approx_exp2(__fmaf_rn(d_qk[33], scale_log2, -row_max0));
                    d_qk[32] = exp2_0;
                    d_qk[33] = exp2_1_0;
                    sb0 += exp2_0 + exp2_1_0;
                    float exp2_2_0 = approx_exp2(__fmaf_rn(d_qk[34], scale_log2, -row_max1));
                    float exp2_3_0 = approx_exp2(__fmaf_rn(d_qk[35], scale_log2, -row_max1));
                    d_qk[34] = exp2_2_0;
                    d_qk[35] = exp2_3_0;
                    sb1 += exp2_2_0 + exp2_3_0;
                    float exp2_4_0 = approx_exp2(__fmaf_rn(d_qk[36], scale_log2, -row_max0));
                    float exp2_5_0 = approx_exp2(__fmaf_rn(d_qk[37], scale_log2, -row_max0));
                    d_qk[36] = exp2_4_0;
                    d_qk[37] = exp2_5_0;
                    sb0 += exp2_4_0 + exp2_5_0;
                    float exp2_6_0 = approx_exp2(__fmaf_rn(d_qk[38], scale_log2, -row_max1));
                    float exp2_7_0 = approx_exp2(__fmaf_rn(d_qk[39], scale_log2, -row_max1));
                    d_qk[38] = exp2_6_0;
                    d_qk[39] = exp2_7_0;
                    sb1 += exp2_6_0 + exp2_7_0;
                    float exp2_8_0 = approx_exp2(__fmaf_rn(d_qk[40], scale_log2, -row_max0));
                    float exp2_9_0 = approx_exp2(__fmaf_rn(d_qk[41], scale_log2, -row_max0));
                    d_qk[40] = exp2_8_0;
                    d_qk[41] = exp2_9_0;
                    sb0 += exp2_8_0 + exp2_9_0;
                    float exp2_10_0 = approx_exp2(__fmaf_rn(d_qk[42], scale_log2, -row_max1));
                    float exp2_11_0 = approx_exp2(__fmaf_rn(d_qk[43], scale_log2, -row_max1));
                    d_qk[42] = exp2_10_0;
                    d_qk[43] = exp2_11_0;
                    sb1 += exp2_10_0 + exp2_11_0;
                    float exp2_12_0 = approx_exp2(__fmaf_rn(d_qk[44], scale_log2, -row_max0));
                    float exp2_13_0 = approx_exp2(__fmaf_rn(d_qk[45], scale_log2, -row_max0));
                    d_qk[44] = exp2_12_0;
                    d_qk[45] = exp2_13_0;
                    sb0 += exp2_12_0 + exp2_13_0;
                    float exp2_14_0 = approx_exp2(__fmaf_rn(d_qk[46], scale_log2, -row_max1));
                    float exp2_15_0 = approx_exp2(__fmaf_rn(d_qk[47], scale_log2, -row_max1));
                    d_qk[46] = exp2_14_0;
                    d_qk[47] = exp2_15_0;
                    sb1 += exp2_14_0 + exp2_15_0;
                    float exp2_16_0 = approx_exp2(__fmaf_rn(d_qk[48], scale_log2, -row_max0));
                    float exp2_17_0 = approx_exp2(__fmaf_rn(d_qk[49], scale_log2, -row_max0));
                    d_qk[48] = exp2_16_0;
                    d_qk[49] = exp2_17_0;
                    sb0 += exp2_16_0 + exp2_17_0;
                    float exp2_18_0 = approx_exp2(__fmaf_rn(d_qk[50], scale_log2, -row_max1));
                    float exp2_19_0 = approx_exp2(__fmaf_rn(d_qk[51], scale_log2, -row_max1));
                    d_qk[50] = exp2_18_0;
                    d_qk[51] = exp2_19_0;
                    sb1 += exp2_18_0 + exp2_19_0;
                    float exp2_20_0 = approx_exp2(__fmaf_rn(d_qk[52], scale_log2, -row_max0));
                    float exp2_21_0 = approx_exp2(__fmaf_rn(d_qk[53], scale_log2, -row_max0));
                    d_qk[52] = exp2_20_0;
                    d_qk[53] = exp2_21_0;
                    sb0 += exp2_20_0 + exp2_21_0;
                    float exp2_22_0 = approx_exp2(__fmaf_rn(d_qk[54], scale_log2, -row_max1));
                    float exp2_23_0 = approx_exp2(__fmaf_rn(d_qk[55], scale_log2, -row_max1));
                    d_qk[54] = exp2_22_0;
                    d_qk[55] = exp2_23_0;
                    sb1 += exp2_22_0 + exp2_23_0;
                    float exp2_24_0 = approx_exp2(__fmaf_rn(d_qk[56], scale_log2, -row_max0));
                    float exp2_25_0 = approx_exp2(__fmaf_rn(d_qk[57], scale_log2, -row_max0));
                    d_qk[56] = exp2_24_0;
                    d_qk[57] = exp2_25_0;
                    sb0 += exp2_24_0 + exp2_25_0;
                    float exp2_26_0 = approx_exp2(__fmaf_rn(d_qk[58], scale_log2, -row_max1));
                    float exp2_27_0 = approx_exp2(__fmaf_rn(d_qk[59], scale_log2, -row_max1));
                    d_qk[58] = exp2_26_0;
                    d_qk[59] = exp2_27_0;
                    sb1 += exp2_26_0 + exp2_27_0;
                    float exp2_28_0 = approx_exp2(__fmaf_rn(d_qk[60], scale_log2, -row_max0));
                    float exp2_29_0 = approx_exp2(__fmaf_rn(d_qk[61], scale_log2, -row_max0));
                    d_qk[60] = exp2_28_0;
                    d_qk[61] = exp2_29_0;
                    sb0 += exp2_28_0 + exp2_29_0;
                    float exp2_30_0 = approx_exp2(__fmaf_rn(d_qk[62], scale_log2, -row_max1));
                    float exp2_31_0 = approx_exp2(__fmaf_rn(d_qk[63], scale_log2, -row_max1));
                    d_qk[62] = exp2_30_0;
                    d_qk[63] = exp2_31_0;
                    sb1 += exp2_30_0 + exp2_31_0;
                    f_sum0 += sa0 + ((has2_0 != 0) ? sb0 : 0.0f);
                    f_sum1 += sa1 + ((has2_0 != 0) ? sb1 : 0.0f);
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 2);
                    f_sum0 += _shfl_xor_sync_26;
                    float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 1);
                    f_sum0 += _shfl_xor_sync_27;
                    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 2);
                    f_sum1 += _shfl_xor_sync_28;
                    float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 1);
                    f_sum1 += _shfl_xor_sync_29;
                    row_sum0 = f_sum0;
                    row_sum1 = f_sum1;
                    __nv_bfloat162 _bit_cast_src_30 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
                    p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_30)[0];
                    __nv_bfloat162 _bit_cast_src_31 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
                    p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_31)[0];
                    __nv_bfloat162 _bit_cast_src_32 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
                    p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_32)[0];
                    __nv_bfloat162 _bit_cast_src_33 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
                    p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_33)[0];
                    __nv_bfloat162 _bit_cast_src_34 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
                    p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_34)[0];
                    __nv_bfloat162 _bit_cast_src_35 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
                    p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
                    __nv_bfloat162 _bit_cast_src_36 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
                    p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
                    __nv_bfloat162 _bit_cast_src_37 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
                    p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
                    __nv_bfloat162 _bit_cast_src_38 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
                    p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
                    __nv_bfloat162 _bit_cast_src_39 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
                    p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
                    __nv_bfloat162 _bit_cast_src_40 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
                    p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
                    __nv_bfloat162 _bit_cast_src_41 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
                    p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
                    __nv_bfloat162 _bit_cast_src_42 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
                    p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
                    __nv_bfloat162 _bit_cast_src_43 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
                    p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
                    __nv_bfloat162 _bit_cast_src_44 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
                    p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
                    __nv_bfloat162 _bit_cast_src_45 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
                    p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
                    __nv_bfloat162 _bit_cast_src_46 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                    p_bf16[16] = reinterpret_cast<unsigned int*>(&_bit_cast_src_46)[0];
                    __nv_bfloat162 _bit_cast_src_47 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                    p_bf16[17] = reinterpret_cast<unsigned int*>(&_bit_cast_src_47)[0];
                    __nv_bfloat162 _bit_cast_src_48 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                    p_bf16[18] = reinterpret_cast<unsigned int*>(&_bit_cast_src_48)[0];
                    __nv_bfloat162 _bit_cast_src_49 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                    p_bf16[19] = reinterpret_cast<unsigned int*>(&_bit_cast_src_49)[0];
                    __nv_bfloat162 _bit_cast_src_50 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                    p_bf16[20] = reinterpret_cast<unsigned int*>(&_bit_cast_src_50)[0];
                    __nv_bfloat162 _bit_cast_src_51 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                    p_bf16[21] = reinterpret_cast<unsigned int*>(&_bit_cast_src_51)[0];
                    __nv_bfloat162 _bit_cast_src_52 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                    p_bf16[22] = reinterpret_cast<unsigned int*>(&_bit_cast_src_52)[0];
                    __nv_bfloat162 _bit_cast_src_53 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                    p_bf16[23] = reinterpret_cast<unsigned int*>(&_bit_cast_src_53)[0];
                    __nv_bfloat162 _bit_cast_src_54 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                    p_bf16[24] = reinterpret_cast<unsigned int*>(&_bit_cast_src_54)[0];
                    __nv_bfloat162 _bit_cast_src_55 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                    p_bf16[25] = reinterpret_cast<unsigned int*>(&_bit_cast_src_55)[0];
                    __nv_bfloat162 _bit_cast_src_56 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                    p_bf16[26] = reinterpret_cast<unsigned int*>(&_bit_cast_src_56)[0];
                    __nv_bfloat162 _bit_cast_src_57 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                    p_bf16[27] = reinterpret_cast<unsigned int*>(&_bit_cast_src_57)[0];
                    __nv_bfloat162 _bit_cast_src_58 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                    p_bf16[28] = reinterpret_cast<unsigned int*>(&_bit_cast_src_58)[0];
                    __nv_bfloat162 _bit_cast_src_59 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                    p_bf16[29] = reinterpret_cast<unsigned int*>(&_bit_cast_src_59)[0];
                    __nv_bfloat162 _bit_cast_src_60 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                    p_bf16[30] = reinterpret_cast<unsigned int*>(&_bit_cast_src_60)[0];
                    __nv_bfloat162 _bit_cast_src_61 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                    p_bf16[31] = reinterpret_cast<unsigned int*>(&_bit_cast_src_61)[0];
                    if (has2_0 == 0) {
                        p_bf16[16] = 0;
                        p_bf16[17] = 0;
                        p_bf16[18] = 0;
                        p_bf16[19] = 0;
                        p_bf16[20] = 0;
                        p_bf16[21] = 0;
                        p_bf16[22] = 0;
                        p_bf16[23] = 0;
                        p_bf16[24] = 0;
                        p_bf16[25] = 0;
                        p_bf16[26] = 0;
                        p_bf16[27] = 0;
                        p_bf16[28] = 0;
                        p_bf16[29] = 0;
                        p_bf16[30] = 0;
                        p_bf16[31] = 0;
                    }
                    float pend0 = 1.0f;
                    float pend1 = 1.0f;
                    int pend_need = 0;
                    int n_loop = n_own - 1;
                    #pragma unroll 1
                    for (int n = 0; n < n_loop; n++) {
                        int cur_stage = cur_pos % 3;
                        int w_e = meta_smem[own_base_w + (n + 1 >> 1)];
                        int en = w_e >> (n + 1 & 1) * 16 & 65535;
                        int nxt_pos = gbase + (en >> 3);
                        int nxt_has2 = en >> 2 & 1;
                        #pragma unroll 1
                        for (int p_3 = cur_pos + 1; p_3 < nxt_pos; p_3++) {
                            int plap_3 = 0;
                            plap_3 += p_3 / 3;
                            int _parity_k_full_4 = (int)((unsigned int)plap_3 % 2);
                            mbarrier_wait(k_full_addr + (p_3 % 3) * 8, _parity_k_full_4);
                            if (elect_sync()) {
                                mbarrier_arrive(k_empty_addr + (p_3 % 3) * 8);
                            }
                            int _parity_v_full_2 = (int)((unsigned int)plap_3 % 2);
                            mbarrier_wait(v_full_addr + (p_3 % 3) * 8, _parity_v_full_2);
                            if (elect_sync()) {
                                mbarrier_arrive(v_empty_addr + (p_3 % 3) * 8);
                            }
                        }
                        int knext = 0;
                        knext += nxt_pos;
                        int knlap = 0;
                        knlap += nxt_pos / 3;
                        int vcur = 0;
                        vcur += cur_pos;
                        int vclap = 0;
                        vclap += cur_pos / 3;
                        int nxt_stage = nxt_pos % 3;
                        int _parity_k_full_5 = (int)((unsigned int)knlap % 2);
                        mbarrier_wait(k_full_addr + (knext % 3) * 8, _parity_k_full_5);
                        int _parity_v_full_3 = (int)((unsigned int)vclap % 2);
                        mbarrier_wait(v_full_addr + (vcur % 3) * 8, _parity_v_full_3);
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_62 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_62 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_62);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(wgmma_b_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(wgmma_b_2 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(wgmma_b_2 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(wgmma_b_2 + 6)
                            : "memory");
                        uint64_t _wgmma_desc_63 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_1_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_63 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_63);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(wgmma_b_1_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(wgmma_b_1_1 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(wgmma_b_1_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(wgmma_b_1_1 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("" : "+f"(d_o[0]) :: "memory");
                        asm volatile("" : "+f"(d_o[1]) :: "memory");
                        asm volatile("" : "+f"(d_o[2]) :: "memory");
                        asm volatile("" : "+f"(d_o[3]) :: "memory");
                        asm volatile("" : "+f"(d_o[4]) :: "memory");
                        asm volatile("" : "+f"(d_o[5]) :: "memory");
                        asm volatile("" : "+f"(d_o[6]) :: "memory");
                        asm volatile("" : "+f"(d_o[7]) :: "memory");
                        asm volatile("" : "+f"(d_o[8]) :: "memory");
                        asm volatile("" : "+f"(d_o[9]) :: "memory");
                        asm volatile("" : "+f"(d_o[10]) :: "memory");
                        asm volatile("" : "+f"(d_o[11]) :: "memory");
                        asm volatile("" : "+f"(d_o[12]) :: "memory");
                        asm volatile("" : "+f"(d_o[13]) :: "memory");
                        asm volatile("" : "+f"(d_o[14]) :: "memory");
                        asm volatile("" : "+f"(d_o[15]) :: "memory");
                        asm volatile("" : "+f"(d_o[16]) :: "memory");
                        asm volatile("" : "+f"(d_o[17]) :: "memory");
                        asm volatile("" : "+f"(d_o[18]) :: "memory");
                        asm volatile("" : "+f"(d_o[19]) :: "memory");
                        asm volatile("" : "+f"(d_o[20]) :: "memory");
                        asm volatile("" : "+f"(d_o[21]) :: "memory");
                        asm volatile("" : "+f"(d_o[22]) :: "memory");
                        asm volatile("" : "+f"(d_o[23]) :: "memory");
                        asm volatile("" : "+f"(d_o[24]) :: "memory");
                        asm volatile("" : "+f"(d_o[25]) :: "memory");
                        asm volatile("" : "+f"(d_o[26]) :: "memory");
                        asm volatile("" : "+f"(d_o[27]) :: "memory");
                        asm volatile("" : "+f"(d_o[28]) :: "memory");
                        asm volatile("" : "+f"(d_o[29]) :: "memory");
                        asm volatile("" : "+f"(d_o[30]) :: "memory");
                        asm volatile("" : "+f"(d_o[31]) :: "memory");
                        asm volatile("" : "+f"(d_o[32]) :: "memory");
                        asm volatile("" : "+f"(d_o[33]) :: "memory");
                        asm volatile("" : "+f"(d_o[34]) :: "memory");
                        asm volatile("" : "+f"(d_o[35]) :: "memory");
                        asm volatile("" : "+f"(d_o[36]) :: "memory");
                        asm volatile("" : "+f"(d_o[37]) :: "memory");
                        asm volatile("" : "+f"(d_o[38]) :: "memory");
                        asm volatile("" : "+f"(d_o[39]) :: "memory");
                        asm volatile("" : "+f"(d_o[40]) :: "memory");
                        asm volatile("" : "+f"(d_o[41]) :: "memory");
                        asm volatile("" : "+f"(d_o[42]) :: "memory");
                        asm volatile("" : "+f"(d_o[43]) :: "memory");
                        asm volatile("" : "+f"(d_o[44]) :: "memory");
                        asm volatile("" : "+f"(d_o[45]) :: "memory");
                        asm volatile("" : "+f"(d_o[46]) :: "memory");
                        asm volatile("" : "+f"(d_o[47]) :: "memory");
                        asm volatile("" : "+f"(d_o[48]) :: "memory");
                        asm volatile("" : "+f"(d_o[49]) :: "memory");
                        asm volatile("" : "+f"(d_o[50]) :: "memory");
                        asm volatile("" : "+f"(d_o[51]) :: "memory");
                        asm volatile("" : "+f"(d_o[52]) :: "memory");
                        asm volatile("" : "+f"(d_o[53]) :: "memory");
                        asm volatile("" : "+f"(d_o[54]) :: "memory");
                        asm volatile("" : "+f"(d_o[55]) :: "memory");
                        asm volatile("" : "+f"(d_o[56]) :: "memory");
                        asm volatile("" : "+f"(d_o[57]) :: "memory");
                        asm volatile("" : "+f"(d_o[58]) :: "memory");
                        asm volatile("" : "+f"(d_o[59]) :: "memory");
                        asm volatile("" : "+f"(d_o[60]) :: "memory");
                        asm volatile("" : "+f"(d_o[61]) :: "memory");
                        asm volatile("" : "+f"(d_o[62]) :: "memory");
                        asm volatile("" : "+f"(d_o[63]) :: "memory");
                        if (pend_need != 0) {
                            d_o[0] = d_o[0] * pend0;
                            d_o[1] = d_o[1] * pend0;
                            d_o[4] = d_o[4] * pend0;
                            d_o[5] = d_o[5] * pend0;
                            d_o[8] = d_o[8] * pend0;
                            d_o[9] = d_o[9] * pend0;
                            d_o[12] = d_o[12] * pend0;
                            d_o[13] = d_o[13] * pend0;
                            d_o[16] = d_o[16] * pend0;
                            d_o[17] = d_o[17] * pend0;
                            d_o[20] = d_o[20] * pend0;
                            d_o[21] = d_o[21] * pend0;
                            d_o[24] = d_o[24] * pend0;
                            d_o[25] = d_o[25] * pend0;
                            d_o[28] = d_o[28] * pend0;
                            d_o[29] = d_o[29] * pend0;
                            d_o[32] = d_o[32] * pend0;
                            d_o[33] = d_o[33] * pend0;
                            d_o[36] = d_o[36] * pend0;
                            d_o[37] = d_o[37] * pend0;
                            d_o[40] = d_o[40] * pend0;
                            d_o[41] = d_o[41] * pend0;
                            d_o[44] = d_o[44] * pend0;
                            d_o[45] = d_o[45] * pend0;
                            d_o[48] = d_o[48] * pend0;
                            d_o[49] = d_o[49] * pend0;
                            d_o[52] = d_o[52] * pend0;
                            d_o[53] = d_o[53] * pend0;
                            d_o[56] = d_o[56] * pend0;
                            d_o[57] = d_o[57] * pend0;
                            d_o[60] = d_o[60] * pend0;
                            d_o[61] = d_o[61] * pend0;
                            d_o[2] = d_o[2] * pend1;
                            d_o[3] = d_o[3] * pend1;
                            d_o[6] = d_o[6] * pend1;
                            d_o[7] = d_o[7] * pend1;
                            d_o[10] = d_o[10] * pend1;
                            d_o[11] = d_o[11] * pend1;
                            d_o[14] = d_o[14] * pend1;
                            d_o[15] = d_o[15] * pend1;
                            d_o[18] = d_o[18] * pend1;
                            d_o[19] = d_o[19] * pend1;
                            d_o[22] = d_o[22] * pend1;
                            d_o[23] = d_o[23] * pend1;
                            d_o[26] = d_o[26] * pend1;
                            d_o[27] = d_o[27] * pend1;
                            d_o[30] = d_o[30] * pend1;
                            d_o[31] = d_o[31] * pend1;
                            d_o[34] = d_o[34] * pend1;
                            d_o[35] = d_o[35] * pend1;
                            d_o[38] = d_o[38] * pend1;
                            d_o[39] = d_o[39] * pend1;
                            d_o[42] = d_o[42] * pend1;
                            d_o[43] = d_o[43] * pend1;
                            d_o[46] = d_o[46] * pend1;
                            d_o[47] = d_o[47] * pend1;
                            d_o[50] = d_o[50] * pend1;
                            d_o[51] = d_o[51] * pend1;
                            d_o[54] = d_o[54] * pend1;
                            d_o[55] = d_o[55] * pend1;
                            d_o[58] = d_o[58] * pend1;
                            d_o[59] = d_o[59] * pend1;
                            d_o[62] = d_o[62] * pend1;
                            d_o[63] = d_o[63] * pend1;
                        }
                        asm volatile("" : "+f"(d_o[0]) :: "memory");
                        asm volatile("" : "+f"(d_o[1]) :: "memory");
                        asm volatile("" : "+f"(d_o[2]) :: "memory");
                        asm volatile("" : "+f"(d_o[3]) :: "memory");
                        asm volatile("" : "+f"(d_o[4]) :: "memory");
                        asm volatile("" : "+f"(d_o[5]) :: "memory");
                        asm volatile("" : "+f"(d_o[6]) :: "memory");
                        asm volatile("" : "+f"(d_o[7]) :: "memory");
                        asm volatile("" : "+f"(d_o[8]) :: "memory");
                        asm volatile("" : "+f"(d_o[9]) :: "memory");
                        asm volatile("" : "+f"(d_o[10]) :: "memory");
                        asm volatile("" : "+f"(d_o[11]) :: "memory");
                        asm volatile("" : "+f"(d_o[12]) :: "memory");
                        asm volatile("" : "+f"(d_o[13]) :: "memory");
                        asm volatile("" : "+f"(d_o[14]) :: "memory");
                        asm volatile("" : "+f"(d_o[15]) :: "memory");
                        asm volatile("" : "+f"(d_o[16]) :: "memory");
                        asm volatile("" : "+f"(d_o[17]) :: "memory");
                        asm volatile("" : "+f"(d_o[18]) :: "memory");
                        asm volatile("" : "+f"(d_o[19]) :: "memory");
                        asm volatile("" : "+f"(d_o[20]) :: "memory");
                        asm volatile("" : "+f"(d_o[21]) :: "memory");
                        asm volatile("" : "+f"(d_o[22]) :: "memory");
                        asm volatile("" : "+f"(d_o[23]) :: "memory");
                        asm volatile("" : "+f"(d_o[24]) :: "memory");
                        asm volatile("" : "+f"(d_o[25]) :: "memory");
                        asm volatile("" : "+f"(d_o[26]) :: "memory");
                        asm volatile("" : "+f"(d_o[27]) :: "memory");
                        asm volatile("" : "+f"(d_o[28]) :: "memory");
                        asm volatile("" : "+f"(d_o[29]) :: "memory");
                        asm volatile("" : "+f"(d_o[30]) :: "memory");
                        asm volatile("" : "+f"(d_o[31]) :: "memory");
                        asm volatile("" : "+f"(d_o[32]) :: "memory");
                        asm volatile("" : "+f"(d_o[33]) :: "memory");
                        asm volatile("" : "+f"(d_o[34]) :: "memory");
                        asm volatile("" : "+f"(d_o[35]) :: "memory");
                        asm volatile("" : "+f"(d_o[36]) :: "memory");
                        asm volatile("" : "+f"(d_o[37]) :: "memory");
                        asm volatile("" : "+f"(d_o[38]) :: "memory");
                        asm volatile("" : "+f"(d_o[39]) :: "memory");
                        asm volatile("" : "+f"(d_o[40]) :: "memory");
                        asm volatile("" : "+f"(d_o[41]) :: "memory");
                        asm volatile("" : "+f"(d_o[42]) :: "memory");
                        asm volatile("" : "+f"(d_o[43]) :: "memory");
                        asm volatile("" : "+f"(d_o[44]) :: "memory");
                        asm volatile("" : "+f"(d_o[45]) :: "memory");
                        asm volatile("" : "+f"(d_o[46]) :: "memory");
                        asm volatile("" : "+f"(d_o[47]) :: "memory");
                        asm volatile("" : "+f"(d_o[48]) :: "memory");
                        asm volatile("" : "+f"(d_o[49]) :: "memory");
                        asm volatile("" : "+f"(d_o[50]) :: "memory");
                        asm volatile("" : "+f"(d_o[51]) :: "memory");
                        asm volatile("" : "+f"(d_o[52]) :: "memory");
                        asm volatile("" : "+f"(d_o[53]) :: "memory");
                        asm volatile("" : "+f"(d_o[54]) :: "memory");
                        asm volatile("" : "+f"(d_o[55]) :: "memory");
                        asm volatile("" : "+f"(d_o[56]) :: "memory");
                        asm volatile("" : "+f"(d_o[57]) :: "memory");
                        asm volatile("" : "+f"(d_o[58]) :: "memory");
                        asm volatile("" : "+f"(d_o[59]) :: "memory");
                        asm volatile("" : "+f"(d_o[60]) :: "memory");
                        asm volatile("" : "+f"(d_o[61]) :: "memory");
                        asm volatile("" : "+f"(d_o[62]) :: "memory");
                        asm volatile("" : "+f"(d_o[63]) :: "memory");
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_64 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_64 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_64);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(wgmma_b_0)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(wgmma_b_0 + 128)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(wgmma_b_0 + 256)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(wgmma_b_0 + 384)
                            : "memory");
                        uint64_t _wgmma_desc_65 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_1_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_65 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_65);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(wgmma_b_1_0)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(wgmma_b_1_0 + 128)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(wgmma_b_1_0 + 256)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(wgmma_b_1_0 + 384)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                        float new_max0 = -CUDART_INF_F;
                        float new_max1 = -CUDART_INF_F;
                        float m0_2 = -CUDART_INF_F;
                        float m1_0 = -CUDART_INF_F;
                        if (scale_log2 >= 0.0f) {
                            float max_1_3 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])), fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]))), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])), fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]))));
                            float max_1_3_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])), fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]))), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])), fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]))));
                            float max_1_4 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37])), fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]))), fmaxf(fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53])), fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]))));
                            float max_1_5 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39])), fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]))), fmaxf(fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55])), fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]))));
                            m0_2 = ((nxt_has2 != 0) ? fmaxf(max_1_3, max_1_4) : max_1_3);
                            m1_0 = ((nxt_has2 != 0) ? fmaxf(max_1_3_1, max_1_5) : max_1_3_1);
                            float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 2);
                            m0_2 = fmaxf(m0_2, _shfl_xor_sync_66);
                            float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 1);
                            m0_2 = fmaxf(m0_2, _shfl_xor_sync_67);
                            float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 2);
                            m1_0 = fmaxf(m1_0, _shfl_xor_sync_68);
                            float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 1);
                            m1_0 = fmaxf(m1_0, _shfl_xor_sync_69);
                        } else {
                            float min_1_3 = fminf(fminf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])), fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]))), fminf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])), fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]))));
                            float min_1_3_1 = fminf(fminf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])), fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]))), fminf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])), fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]))));
                            float min_1_4 = fminf(fminf(fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37])), fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]))), fminf(fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53])), fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]))));
                            float min_1_5 = fminf(fminf(fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39])), fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]))), fminf(fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55])), fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]))));
                            m0_2 = ((nxt_has2 != 0) ? fminf(min_1_3, min_1_4) : min_1_3);
                            m1_0 = ((nxt_has2 != 0) ? fminf(min_1_3_1, min_1_5) : min_1_3_1);
                            float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 2);
                            m0_2 = fminf(m0_2, _shfl_xor_sync_70);
                            float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, m0_2, 1);
                            m0_2 = fminf(m0_2, _shfl_xor_sync_71);
                            float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 2);
                            m1_0 = fminf(m1_0, _shfl_xor_sync_72);
                            float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 1);
                            m1_0 = fminf(m1_0, _shfl_xor_sync_73);
                        }
                        new_max0 = m0_2;
                        new_max1 = m1_0;
                        float merged_max0 = new_max0 * scale_log2;
                        float merged_max1 = new_max1 * scale_log2;
                        merged_max0 = fmaxf(merged_max0, row_max0);
                        merged_max1 = fmaxf(merged_max1, row_max1);
                        float _exp2_64 = approx_exp2(row_max0 - merged_max0);
                        float _exp2_65 = approx_exp2(row_max1 - merged_max1);
                        row_max0 = merged_max0;
                        row_max1 = merged_max1;
                        float new_sum0 = 0.0f;
                        float new_sum1 = 0.0f;
                        float sa0_0 = 0.0f;
                        float sa1_0 = 0.0f;
                        float sb0_0 = 0.0f;
                        float sb1_0 = 0.0f;
                        float exp2_32 = approx_exp2(__fmaf_rn(d_qk[0], scale_log2, -merged_max0));
                        float exp2_1_1 = approx_exp2(__fmaf_rn(d_qk[1], scale_log2, -merged_max0));
                        d_qk[0] = exp2_32;
                        d_qk[1] = exp2_1_1;
                        sa0_0 += exp2_32 + exp2_1_1;
                        float exp2_2_1 = approx_exp2(__fmaf_rn(d_qk[2], scale_log2, -merged_max1));
                        float exp2_3_1 = approx_exp2(__fmaf_rn(d_qk[3], scale_log2, -merged_max1));
                        d_qk[2] = exp2_2_1;
                        d_qk[3] = exp2_3_1;
                        sa1_0 += exp2_2_1 + exp2_3_1;
                        float exp2_4_1 = approx_exp2(__fmaf_rn(d_qk[4], scale_log2, -merged_max0));
                        float exp2_5_1 = approx_exp2(__fmaf_rn(d_qk[5], scale_log2, -merged_max0));
                        d_qk[4] = exp2_4_1;
                        d_qk[5] = exp2_5_1;
                        sa0_0 += exp2_4_1 + exp2_5_1;
                        float exp2_6_1 = approx_exp2(__fmaf_rn(d_qk[6], scale_log2, -merged_max1));
                        float exp2_7_1 = approx_exp2(__fmaf_rn(d_qk[7], scale_log2, -merged_max1));
                        d_qk[6] = exp2_6_1;
                        d_qk[7] = exp2_7_1;
                        sa1_0 += exp2_6_1 + exp2_7_1;
                        float exp2_8_1 = approx_exp2(__fmaf_rn(d_qk[8], scale_log2, -merged_max0));
                        float exp2_9_1 = approx_exp2(__fmaf_rn(d_qk[9], scale_log2, -merged_max0));
                        d_qk[8] = exp2_8_1;
                        d_qk[9] = exp2_9_1;
                        sa0_0 += exp2_8_1 + exp2_9_1;
                        float exp2_10_1 = approx_exp2(__fmaf_rn(d_qk[10], scale_log2, -merged_max1));
                        float exp2_11_1 = approx_exp2(__fmaf_rn(d_qk[11], scale_log2, -merged_max1));
                        d_qk[10] = exp2_10_1;
                        d_qk[11] = exp2_11_1;
                        sa1_0 += exp2_10_1 + exp2_11_1;
                        float exp2_12_1 = approx_exp2(__fmaf_rn(d_qk[12], scale_log2, -merged_max0));
                        float exp2_13_1 = approx_exp2(__fmaf_rn(d_qk[13], scale_log2, -merged_max0));
                        d_qk[12] = exp2_12_1;
                        d_qk[13] = exp2_13_1;
                        sa0_0 += exp2_12_1 + exp2_13_1;
                        float exp2_14_1 = approx_exp2(__fmaf_rn(d_qk[14], scale_log2, -merged_max1));
                        float exp2_15_1 = approx_exp2(__fmaf_rn(d_qk[15], scale_log2, -merged_max1));
                        d_qk[14] = exp2_14_1;
                        d_qk[15] = exp2_15_1;
                        sa1_0 += exp2_14_1 + exp2_15_1;
                        float exp2_16_1 = approx_exp2(__fmaf_rn(d_qk[16], scale_log2, -merged_max0));
                        float exp2_17_1 = approx_exp2(__fmaf_rn(d_qk[17], scale_log2, -merged_max0));
                        d_qk[16] = exp2_16_1;
                        d_qk[17] = exp2_17_1;
                        sa0_0 += exp2_16_1 + exp2_17_1;
                        float exp2_18_1 = approx_exp2(__fmaf_rn(d_qk[18], scale_log2, -merged_max1));
                        float exp2_19_1 = approx_exp2(__fmaf_rn(d_qk[19], scale_log2, -merged_max1));
                        d_qk[18] = exp2_18_1;
                        d_qk[19] = exp2_19_1;
                        sa1_0 += exp2_18_1 + exp2_19_1;
                        float exp2_20_1 = approx_exp2(__fmaf_rn(d_qk[20], scale_log2, -merged_max0));
                        float exp2_21_1 = approx_exp2(__fmaf_rn(d_qk[21], scale_log2, -merged_max0));
                        d_qk[20] = exp2_20_1;
                        d_qk[21] = exp2_21_1;
                        sa0_0 += exp2_20_1 + exp2_21_1;
                        float exp2_22_1 = approx_exp2(__fmaf_rn(d_qk[22], scale_log2, -merged_max1));
                        float exp2_23_1 = approx_exp2(__fmaf_rn(d_qk[23], scale_log2, -merged_max1));
                        d_qk[22] = exp2_22_1;
                        d_qk[23] = exp2_23_1;
                        sa1_0 += exp2_22_1 + exp2_23_1;
                        float exp2_24_1 = approx_exp2(__fmaf_rn(d_qk[24], scale_log2, -merged_max0));
                        float exp2_25_1 = approx_exp2(__fmaf_rn(d_qk[25], scale_log2, -merged_max0));
                        d_qk[24] = exp2_24_1;
                        d_qk[25] = exp2_25_1;
                        sa0_0 += exp2_24_1 + exp2_25_1;
                        float exp2_26_1 = approx_exp2(__fmaf_rn(d_qk[26], scale_log2, -merged_max1));
                        float exp2_27_1 = approx_exp2(__fmaf_rn(d_qk[27], scale_log2, -merged_max1));
                        d_qk[26] = exp2_26_1;
                        d_qk[27] = exp2_27_1;
                        sa1_0 += exp2_26_1 + exp2_27_1;
                        float exp2_28_1 = approx_exp2(__fmaf_rn(d_qk[28], scale_log2, -merged_max0));
                        float exp2_29_1 = approx_exp2(__fmaf_rn(d_qk[29], scale_log2, -merged_max0));
                        d_qk[28] = exp2_28_1;
                        d_qk[29] = exp2_29_1;
                        sa0_0 += exp2_28_1 + exp2_29_1;
                        float exp2_30_1 = approx_exp2(__fmaf_rn(d_qk[30], scale_log2, -merged_max1));
                        float exp2_31_1 = approx_exp2(__fmaf_rn(d_qk[31], scale_log2, -merged_max1));
                        d_qk[30] = exp2_30_1;
                        d_qk[31] = exp2_31_1;
                        sa1_0 += exp2_30_1 + exp2_31_1;
                        float exp2_33 = approx_exp2(__fmaf_rn(d_qk[32], scale_log2, -merged_max0));
                        float exp2_1_2 = approx_exp2(__fmaf_rn(d_qk[33], scale_log2, -merged_max0));
                        d_qk[32] = exp2_33;
                        d_qk[33] = exp2_1_2;
                        sb0_0 += exp2_33 + exp2_1_2;
                        float exp2_2_2 = approx_exp2(__fmaf_rn(d_qk[34], scale_log2, -merged_max1));
                        float exp2_3_2 = approx_exp2(__fmaf_rn(d_qk[35], scale_log2, -merged_max1));
                        d_qk[34] = exp2_2_2;
                        d_qk[35] = exp2_3_2;
                        sb1_0 += exp2_2_2 + exp2_3_2;
                        float exp2_4_2 = approx_exp2(__fmaf_rn(d_qk[36], scale_log2, -merged_max0));
                        float exp2_5_2 = approx_exp2(__fmaf_rn(d_qk[37], scale_log2, -merged_max0));
                        d_qk[36] = exp2_4_2;
                        d_qk[37] = exp2_5_2;
                        sb0_0 += exp2_4_2 + exp2_5_2;
                        float exp2_6_2 = approx_exp2(__fmaf_rn(d_qk[38], scale_log2, -merged_max1));
                        float exp2_7_2 = approx_exp2(__fmaf_rn(d_qk[39], scale_log2, -merged_max1));
                        d_qk[38] = exp2_6_2;
                        d_qk[39] = exp2_7_2;
                        sb1_0 += exp2_6_2 + exp2_7_2;
                        float exp2_8_2 = approx_exp2(__fmaf_rn(d_qk[40], scale_log2, -merged_max0));
                        float exp2_9_2 = approx_exp2(__fmaf_rn(d_qk[41], scale_log2, -merged_max0));
                        d_qk[40] = exp2_8_2;
                        d_qk[41] = exp2_9_2;
                        sb0_0 += exp2_8_2 + exp2_9_2;
                        float exp2_10_2 = approx_exp2(__fmaf_rn(d_qk[42], scale_log2, -merged_max1));
                        float exp2_11_2 = approx_exp2(__fmaf_rn(d_qk[43], scale_log2, -merged_max1));
                        d_qk[42] = exp2_10_2;
                        d_qk[43] = exp2_11_2;
                        sb1_0 += exp2_10_2 + exp2_11_2;
                        float exp2_12_2 = approx_exp2(__fmaf_rn(d_qk[44], scale_log2, -merged_max0));
                        float exp2_13_2 = approx_exp2(__fmaf_rn(d_qk[45], scale_log2, -merged_max0));
                        d_qk[44] = exp2_12_2;
                        d_qk[45] = exp2_13_2;
                        sb0_0 += exp2_12_2 + exp2_13_2;
                        float exp2_14_2 = approx_exp2(__fmaf_rn(d_qk[46], scale_log2, -merged_max1));
                        float exp2_15_2 = approx_exp2(__fmaf_rn(d_qk[47], scale_log2, -merged_max1));
                        d_qk[46] = exp2_14_2;
                        d_qk[47] = exp2_15_2;
                        sb1_0 += exp2_14_2 + exp2_15_2;
                        float exp2_16_2 = approx_exp2(__fmaf_rn(d_qk[48], scale_log2, -merged_max0));
                        float exp2_17_2 = approx_exp2(__fmaf_rn(d_qk[49], scale_log2, -merged_max0));
                        d_qk[48] = exp2_16_2;
                        d_qk[49] = exp2_17_2;
                        sb0_0 += exp2_16_2 + exp2_17_2;
                        float exp2_18_2 = approx_exp2(__fmaf_rn(d_qk[50], scale_log2, -merged_max1));
                        float exp2_19_2 = approx_exp2(__fmaf_rn(d_qk[51], scale_log2, -merged_max1));
                        d_qk[50] = exp2_18_2;
                        d_qk[51] = exp2_19_2;
                        sb1_0 += exp2_18_2 + exp2_19_2;
                        float exp2_20_2 = approx_exp2(__fmaf_rn(d_qk[52], scale_log2, -merged_max0));
                        float exp2_21_2 = approx_exp2(__fmaf_rn(d_qk[53], scale_log2, -merged_max0));
                        d_qk[52] = exp2_20_2;
                        d_qk[53] = exp2_21_2;
                        sb0_0 += exp2_20_2 + exp2_21_2;
                        float exp2_22_2 = approx_exp2(__fmaf_rn(d_qk[54], scale_log2, -merged_max1));
                        float exp2_23_2 = approx_exp2(__fmaf_rn(d_qk[55], scale_log2, -merged_max1));
                        d_qk[54] = exp2_22_2;
                        d_qk[55] = exp2_23_2;
                        sb1_0 += exp2_22_2 + exp2_23_2;
                        float exp2_24_2 = approx_exp2(__fmaf_rn(d_qk[56], scale_log2, -merged_max0));
                        float exp2_25_2 = approx_exp2(__fmaf_rn(d_qk[57], scale_log2, -merged_max0));
                        d_qk[56] = exp2_24_2;
                        d_qk[57] = exp2_25_2;
                        sb0_0 += exp2_24_2 + exp2_25_2;
                        float exp2_26_2 = approx_exp2(__fmaf_rn(d_qk[58], scale_log2, -merged_max1));
                        float exp2_27_2 = approx_exp2(__fmaf_rn(d_qk[59], scale_log2, -merged_max1));
                        d_qk[58] = exp2_26_2;
                        d_qk[59] = exp2_27_2;
                        sb1_0 += exp2_26_2 + exp2_27_2;
                        float exp2_28_2 = approx_exp2(__fmaf_rn(d_qk[60], scale_log2, -merged_max0));
                        float exp2_29_2 = approx_exp2(__fmaf_rn(d_qk[61], scale_log2, -merged_max0));
                        d_qk[60] = exp2_28_2;
                        d_qk[61] = exp2_29_2;
                        sb0_0 += exp2_28_2 + exp2_29_2;
                        float exp2_30_2 = approx_exp2(__fmaf_rn(d_qk[62], scale_log2, -merged_max1));
                        float exp2_31_2 = approx_exp2(__fmaf_rn(d_qk[63], scale_log2, -merged_max1));
                        d_qk[62] = exp2_30_2;
                        d_qk[63] = exp2_31_2;
                        sb1_0 += exp2_30_2 + exp2_31_2;
                        new_sum0 += sa0_0 + ((nxt_has2 != 0) ? sb0_0 : 0.0f);
                        new_sum1 += sa1_0 + ((nxt_has2 != 0) ? sb1_0 : 0.0f);
                        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 2);
                        new_sum0 += _shfl_xor_sync_74;
                        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 1);
                        new_sum0 += _shfl_xor_sync_75;
                        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 2);
                        new_sum1 += _shfl_xor_sync_76;
                        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 1);
                        new_sum1 += _shfl_xor_sync_77;
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (knext % 3) * 8);
                        }
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (vcur % 3) * 8);
                        }
                        pend0 = _exp2_64;
                        pend1 = _exp2_65;
                        int _warp_vote_78 = __any_sync(0xFFFFFFFF, _exp2_64 != 1.0f || _exp2_65 != 1.0f);
                        pend_need = _warp_vote_78;
                        row_sum0 = __fmaf_rn(row_sum0, _exp2_64, new_sum0);
                        row_sum1 = __fmaf_rn(row_sum1, _exp2_65, new_sum1);
                        __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
                        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
                        __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
                        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
                        __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
                        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
                        __nv_bfloat162 _bit_cast_src_82 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
                        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_82)[0];
                        __nv_bfloat162 _bit_cast_src_83 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
                        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_83)[0];
                        __nv_bfloat162 _bit_cast_src_84 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
                        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_84)[0];
                        __nv_bfloat162 _bit_cast_src_85 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
                        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_85)[0];
                        __nv_bfloat162 _bit_cast_src_86 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
                        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
                        __nv_bfloat162 _bit_cast_src_87 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
                        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
                        __nv_bfloat162 _bit_cast_src_88 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
                        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
                        __nv_bfloat162 _bit_cast_src_89 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
                        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_89)[0];
                        __nv_bfloat162 _bit_cast_src_90 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
                        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_90)[0];
                        __nv_bfloat162 _bit_cast_src_91 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
                        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_91)[0];
                        __nv_bfloat162 _bit_cast_src_92 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
                        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_92)[0];
                        __nv_bfloat162 _bit_cast_src_93 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
                        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_93)[0];
                        __nv_bfloat162 _bit_cast_src_94 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
                        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_94)[0];
                        __nv_bfloat162 _bit_cast_src_95 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                        p_bf16[16] = reinterpret_cast<unsigned int*>(&_bit_cast_src_95)[0];
                        __nv_bfloat162 _bit_cast_src_96 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                        p_bf16[17] = reinterpret_cast<unsigned int*>(&_bit_cast_src_96)[0];
                        __nv_bfloat162 _bit_cast_src_97 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                        p_bf16[18] = reinterpret_cast<unsigned int*>(&_bit_cast_src_97)[0];
                        __nv_bfloat162 _bit_cast_src_98 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                        p_bf16[19] = reinterpret_cast<unsigned int*>(&_bit_cast_src_98)[0];
                        __nv_bfloat162 _bit_cast_src_99 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                        p_bf16[20] = reinterpret_cast<unsigned int*>(&_bit_cast_src_99)[0];
                        __nv_bfloat162 _bit_cast_src_100 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                        p_bf16[21] = reinterpret_cast<unsigned int*>(&_bit_cast_src_100)[0];
                        __nv_bfloat162 _bit_cast_src_101 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                        p_bf16[22] = reinterpret_cast<unsigned int*>(&_bit_cast_src_101)[0];
                        __nv_bfloat162 _bit_cast_src_102 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                        p_bf16[23] = reinterpret_cast<unsigned int*>(&_bit_cast_src_102)[0];
                        __nv_bfloat162 _bit_cast_src_103 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                        p_bf16[24] = reinterpret_cast<unsigned int*>(&_bit_cast_src_103)[0];
                        __nv_bfloat162 _bit_cast_src_104 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                        p_bf16[25] = reinterpret_cast<unsigned int*>(&_bit_cast_src_104)[0];
                        __nv_bfloat162 _bit_cast_src_105 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                        p_bf16[26] = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
                        __nv_bfloat162 _bit_cast_src_106 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                        p_bf16[27] = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
                        __nv_bfloat162 _bit_cast_src_107 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                        p_bf16[28] = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
                        __nv_bfloat162 _bit_cast_src_108 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                        p_bf16[29] = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
                        __nv_bfloat162 _bit_cast_src_109 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                        p_bf16[30] = reinterpret_cast<unsigned int*>(&_bit_cast_src_109)[0];
                        __nv_bfloat162 _bit_cast_src_110 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                        p_bf16[31] = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
                        if (nxt_has2 == 0) {
                            p_bf16[16] = 0;
                            p_bf16[17] = 0;
                            p_bf16[18] = 0;
                            p_bf16[19] = 0;
                            p_bf16[20] = 0;
                            p_bf16[21] = 0;
                            p_bf16[22] = 0;
                            p_bf16[23] = 0;
                            p_bf16[24] = 0;
                            p_bf16[25] = 0;
                            p_bf16[26] = 0;
                            p_bf16[27] = 0;
                            p_bf16[28] = 0;
                            p_bf16[29] = 0;
                            p_bf16[30] = 0;
                            p_bf16[31] = 0;
                        }
                        prev = nxt_pos;
                        cur_pos = nxt_pos;
                        cur_has2 = nxt_has2;
                    }
                    int last_stage = cur_pos % 3;
                    int vlast = 0;
                    vlast += cur_pos;
                    int vllap = 0;
                    vllap += cur_pos / 3;
                    int _parity_v_full_4 = (int)((unsigned int)vllap % 2);
                    mbarrier_wait(v_full_addr + (vlast % 3) * 8, _parity_v_full_4);
                    asm volatile("" : "+f"(d_o[0]) :: "memory");
                    asm volatile("" : "+f"(d_o[1]) :: "memory");
                    asm volatile("" : "+f"(d_o[2]) :: "memory");
                    asm volatile("" : "+f"(d_o[3]) :: "memory");
                    asm volatile("" : "+f"(d_o[4]) :: "memory");
                    asm volatile("" : "+f"(d_o[5]) :: "memory");
                    asm volatile("" : "+f"(d_o[6]) :: "memory");
                    asm volatile("" : "+f"(d_o[7]) :: "memory");
                    asm volatile("" : "+f"(d_o[8]) :: "memory");
                    asm volatile("" : "+f"(d_o[9]) :: "memory");
                    asm volatile("" : "+f"(d_o[10]) :: "memory");
                    asm volatile("" : "+f"(d_o[11]) :: "memory");
                    asm volatile("" : "+f"(d_o[12]) :: "memory");
                    asm volatile("" : "+f"(d_o[13]) :: "memory");
                    asm volatile("" : "+f"(d_o[14]) :: "memory");
                    asm volatile("" : "+f"(d_o[15]) :: "memory");
                    asm volatile("" : "+f"(d_o[16]) :: "memory");
                    asm volatile("" : "+f"(d_o[17]) :: "memory");
                    asm volatile("" : "+f"(d_o[18]) :: "memory");
                    asm volatile("" : "+f"(d_o[19]) :: "memory");
                    asm volatile("" : "+f"(d_o[20]) :: "memory");
                    asm volatile("" : "+f"(d_o[21]) :: "memory");
                    asm volatile("" : "+f"(d_o[22]) :: "memory");
                    asm volatile("" : "+f"(d_o[23]) :: "memory");
                    asm volatile("" : "+f"(d_o[24]) :: "memory");
                    asm volatile("" : "+f"(d_o[25]) :: "memory");
                    asm volatile("" : "+f"(d_o[26]) :: "memory");
                    asm volatile("" : "+f"(d_o[27]) :: "memory");
                    asm volatile("" : "+f"(d_o[28]) :: "memory");
                    asm volatile("" : "+f"(d_o[29]) :: "memory");
                    asm volatile("" : "+f"(d_o[30]) :: "memory");
                    asm volatile("" : "+f"(d_o[31]) :: "memory");
                    asm volatile("" : "+f"(d_o[32]) :: "memory");
                    asm volatile("" : "+f"(d_o[33]) :: "memory");
                    asm volatile("" : "+f"(d_o[34]) :: "memory");
                    asm volatile("" : "+f"(d_o[35]) :: "memory");
                    asm volatile("" : "+f"(d_o[36]) :: "memory");
                    asm volatile("" : "+f"(d_o[37]) :: "memory");
                    asm volatile("" : "+f"(d_o[38]) :: "memory");
                    asm volatile("" : "+f"(d_o[39]) :: "memory");
                    asm volatile("" : "+f"(d_o[40]) :: "memory");
                    asm volatile("" : "+f"(d_o[41]) :: "memory");
                    asm volatile("" : "+f"(d_o[42]) :: "memory");
                    asm volatile("" : "+f"(d_o[43]) :: "memory");
                    asm volatile("" : "+f"(d_o[44]) :: "memory");
                    asm volatile("" : "+f"(d_o[45]) :: "memory");
                    asm volatile("" : "+f"(d_o[46]) :: "memory");
                    asm volatile("" : "+f"(d_o[47]) :: "memory");
                    asm volatile("" : "+f"(d_o[48]) :: "memory");
                    asm volatile("" : "+f"(d_o[49]) :: "memory");
                    asm volatile("" : "+f"(d_o[50]) :: "memory");
                    asm volatile("" : "+f"(d_o[51]) :: "memory");
                    asm volatile("" : "+f"(d_o[52]) :: "memory");
                    asm volatile("" : "+f"(d_o[53]) :: "memory");
                    asm volatile("" : "+f"(d_o[54]) :: "memory");
                    asm volatile("" : "+f"(d_o[55]) :: "memory");
                    asm volatile("" : "+f"(d_o[56]) :: "memory");
                    asm volatile("" : "+f"(d_o[57]) :: "memory");
                    asm volatile("" : "+f"(d_o[58]) :: "memory");
                    asm volatile("" : "+f"(d_o[59]) :: "memory");
                    asm volatile("" : "+f"(d_o[60]) :: "memory");
                    asm volatile("" : "+f"(d_o[61]) :: "memory");
                    asm volatile("" : "+f"(d_o[62]) :: "memory");
                    asm volatile("" : "+f"(d_o[63]) :: "memory");
                    if (pend_need != 0) {
                        d_o[0] = d_o[0] * pend0;
                        d_o[1] = d_o[1] * pend0;
                        d_o[4] = d_o[4] * pend0;
                        d_o[5] = d_o[5] * pend0;
                        d_o[8] = d_o[8] * pend0;
                        d_o[9] = d_o[9] * pend0;
                        d_o[12] = d_o[12] * pend0;
                        d_o[13] = d_o[13] * pend0;
                        d_o[16] = d_o[16] * pend0;
                        d_o[17] = d_o[17] * pend0;
                        d_o[20] = d_o[20] * pend0;
                        d_o[21] = d_o[21] * pend0;
                        d_o[24] = d_o[24] * pend0;
                        d_o[25] = d_o[25] * pend0;
                        d_o[28] = d_o[28] * pend0;
                        d_o[29] = d_o[29] * pend0;
                        d_o[32] = d_o[32] * pend0;
                        d_o[33] = d_o[33] * pend0;
                        d_o[36] = d_o[36] * pend0;
                        d_o[37] = d_o[37] * pend0;
                        d_o[40] = d_o[40] * pend0;
                        d_o[41] = d_o[41] * pend0;
                        d_o[44] = d_o[44] * pend0;
                        d_o[45] = d_o[45] * pend0;
                        d_o[48] = d_o[48] * pend0;
                        d_o[49] = d_o[49] * pend0;
                        d_o[52] = d_o[52] * pend0;
                        d_o[53] = d_o[53] * pend0;
                        d_o[56] = d_o[56] * pend0;
                        d_o[57] = d_o[57] * pend0;
                        d_o[60] = d_o[60] * pend0;
                        d_o[61] = d_o[61] * pend0;
                        d_o[2] = d_o[2] * pend1;
                        d_o[3] = d_o[3] * pend1;
                        d_o[6] = d_o[6] * pend1;
                        d_o[7] = d_o[7] * pend1;
                        d_o[10] = d_o[10] * pend1;
                        d_o[11] = d_o[11] * pend1;
                        d_o[14] = d_o[14] * pend1;
                        d_o[15] = d_o[15] * pend1;
                        d_o[18] = d_o[18] * pend1;
                        d_o[19] = d_o[19] * pend1;
                        d_o[22] = d_o[22] * pend1;
                        d_o[23] = d_o[23] * pend1;
                        d_o[26] = d_o[26] * pend1;
                        d_o[27] = d_o[27] * pend1;
                        d_o[30] = d_o[30] * pend1;
                        d_o[31] = d_o[31] * pend1;
                        d_o[34] = d_o[34] * pend1;
                        d_o[35] = d_o[35] * pend1;
                        d_o[38] = d_o[38] * pend1;
                        d_o[39] = d_o[39] * pend1;
                        d_o[42] = d_o[42] * pend1;
                        d_o[43] = d_o[43] * pend1;
                        d_o[46] = d_o[46] * pend1;
                        d_o[47] = d_o[47] * pend1;
                        d_o[50] = d_o[50] * pend1;
                        d_o[51] = d_o[51] * pend1;
                        d_o[54] = d_o[54] * pend1;
                        d_o[55] = d_o[55] * pend1;
                        d_o[58] = d_o[58] * pend1;
                        d_o[59] = d_o[59] * pend1;
                        d_o[62] = d_o[62] * pend1;
                        d_o[63] = d_o[63] * pend1;
                    }
                    asm volatile("" : "+f"(d_o[0]) :: "memory");
                    asm volatile("" : "+f"(d_o[1]) :: "memory");
                    asm volatile("" : "+f"(d_o[2]) :: "memory");
                    asm volatile("" : "+f"(d_o[3]) :: "memory");
                    asm volatile("" : "+f"(d_o[4]) :: "memory");
                    asm volatile("" : "+f"(d_o[5]) :: "memory");
                    asm volatile("" : "+f"(d_o[6]) :: "memory");
                    asm volatile("" : "+f"(d_o[7]) :: "memory");
                    asm volatile("" : "+f"(d_o[8]) :: "memory");
                    asm volatile("" : "+f"(d_o[9]) :: "memory");
                    asm volatile("" : "+f"(d_o[10]) :: "memory");
                    asm volatile("" : "+f"(d_o[11]) :: "memory");
                    asm volatile("" : "+f"(d_o[12]) :: "memory");
                    asm volatile("" : "+f"(d_o[13]) :: "memory");
                    asm volatile("" : "+f"(d_o[14]) :: "memory");
                    asm volatile("" : "+f"(d_o[15]) :: "memory");
                    asm volatile("" : "+f"(d_o[16]) :: "memory");
                    asm volatile("" : "+f"(d_o[17]) :: "memory");
                    asm volatile("" : "+f"(d_o[18]) :: "memory");
                    asm volatile("" : "+f"(d_o[19]) :: "memory");
                    asm volatile("" : "+f"(d_o[20]) :: "memory");
                    asm volatile("" : "+f"(d_o[21]) :: "memory");
                    asm volatile("" : "+f"(d_o[22]) :: "memory");
                    asm volatile("" : "+f"(d_o[23]) :: "memory");
                    asm volatile("" : "+f"(d_o[24]) :: "memory");
                    asm volatile("" : "+f"(d_o[25]) :: "memory");
                    asm volatile("" : "+f"(d_o[26]) :: "memory");
                    asm volatile("" : "+f"(d_o[27]) :: "memory");
                    asm volatile("" : "+f"(d_o[28]) :: "memory");
                    asm volatile("" : "+f"(d_o[29]) :: "memory");
                    asm volatile("" : "+f"(d_o[30]) :: "memory");
                    asm volatile("" : "+f"(d_o[31]) :: "memory");
                    asm volatile("" : "+f"(d_o[32]) :: "memory");
                    asm volatile("" : "+f"(d_o[33]) :: "memory");
                    asm volatile("" : "+f"(d_o[34]) :: "memory");
                    asm volatile("" : "+f"(d_o[35]) :: "memory");
                    asm volatile("" : "+f"(d_o[36]) :: "memory");
                    asm volatile("" : "+f"(d_o[37]) :: "memory");
                    asm volatile("" : "+f"(d_o[38]) :: "memory");
                    asm volatile("" : "+f"(d_o[39]) :: "memory");
                    asm volatile("" : "+f"(d_o[40]) :: "memory");
                    asm volatile("" : "+f"(d_o[41]) :: "memory");
                    asm volatile("" : "+f"(d_o[42]) :: "memory");
                    asm volatile("" : "+f"(d_o[43]) :: "memory");
                    asm volatile("" : "+f"(d_o[44]) :: "memory");
                    asm volatile("" : "+f"(d_o[45]) :: "memory");
                    asm volatile("" : "+f"(d_o[46]) :: "memory");
                    asm volatile("" : "+f"(d_o[47]) :: "memory");
                    asm volatile("" : "+f"(d_o[48]) :: "memory");
                    asm volatile("" : "+f"(d_o[49]) :: "memory");
                    asm volatile("" : "+f"(d_o[50]) :: "memory");
                    asm volatile("" : "+f"(d_o[51]) :: "memory");
                    asm volatile("" : "+f"(d_o[52]) :: "memory");
                    asm volatile("" : "+f"(d_o[53]) :: "memory");
                    asm volatile("" : "+f"(d_o[54]) :: "memory");
                    asm volatile("" : "+f"(d_o[55]) :: "memory");
                    asm volatile("" : "+f"(d_o[56]) :: "memory");
                    asm volatile("" : "+f"(d_o[57]) :: "memory");
                    asm volatile("" : "+f"(d_o[58]) :: "memory");
                    asm volatile("" : "+f"(d_o[59]) :: "memory");
                    asm volatile("" : "+f"(d_o[60]) :: "memory");
                    asm volatile("" : "+f"(d_o[61]) :: "memory");
                    asm volatile("" : "+f"(d_o[62]) :: "memory");
                    asm volatile("" : "+f"(d_o[63]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_111 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_111 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_111);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(wgmma_b_3)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(wgmma_b_3 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(wgmma_b_3 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(wgmma_b_3 + 384)
                        : "memory");
                    uint64_t _wgmma_desc_112 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b_1_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_112 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_112);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(wgmma_b_1_2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(wgmma_b_1_2 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(wgmma_b_1_2 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(wgmma_b_1_2 + 384)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (vlast % 3) * 8);
                    }
                    prev = cur_pos;
                }
                int tile_end = gbase + n_seq_m;
                int tail_lo = prev + 1;
                #pragma unroll 1
                for (int p_4 = tail_lo; p_4 < tile_end; p_4++) {
                    int tlap = 0;
                    tlap += p_4 / 3;
                    int _parity_k_full_6 = (int)((unsigned int)tlap % 2);
                    mbarrier_wait(k_full_addr + (p_4 % 3) * 8, _parity_k_full_6);
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (p_4 % 3) * 8);
                    }
                    int _parity_v_full_5 = (int)((unsigned int)tlap % 2);
                    mbarrier_wait(v_full_addr + (p_4 % 3) * 8, _parity_v_full_5);
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (p_4 % 3) * 8);
                    }
                }
                prev = tile_end - 1;
                gbase = tile_end;
                int ovl_m = n_seq_m >= 3;
                nx[7] = 0;
                nx[8] = gbase;
                nx[9] = 0;
                nx[10] = 0;
                if (ovl_m != 0) {
                    nx[10] = 1;
                    int slot_n_1 = ti_3 + 1 & 1;
                    int mbase_n_1 = slot_n_1 * 144;
                    nx[7] = 0;
                    int _stage_meta_full_3 = (int)((unsigned int)(ti_3 + 1) % 2);
                    int _parity_meta_full_3 = (int)((unsigned int)(ti_3 + 1) / 2 % 2);
                    mbarrier_wait(meta_full_addr + (_stage_meta_full_3) * 8, _parity_meta_full_3);
                    int valid_n_1 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 7], 0);
                    if (valid_n_1 != 0) {
                        nx[7] = 1;
                    }
                    if (nx[7] != 0) {
                        int _shfl_sync_113 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1], 0);
                        nx[0] = _shfl_sync_113;
                        int _shfl_sync_114 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 1], 0);
                        nx[1] = _shfl_sync_114;
                        int _shfl_sync_115 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 2], 0);
                        nx[2] = _shfl_sync_115;
                        int _shfl_sync_116 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 3], 0);
                        nx[3] = _shfl_sync_116;
                        int _shfl_sync_117 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 4], 0);
                        nx[4] = _shfl_sync_117;
                        int _shfl_sync_118 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 5 + cwg], 0);
                        nx[5] = _shfl_sync_118;
                        int _shfl_sync_119 = __shfl_sync(0xFFFFFFFF, meta_smem[mbase_n_1 + 5 + 1 - cwg], 0);
                        nx[6] = _shfl_sync_119;
                    }
                    if (nx[7] != 0) {
                        int qs_n_2 = ((nx[3] == 1) ? 0 : cwg);
                        int _parity_q_ready_1 = (int)((unsigned int)ti_3 % 2) ^ 1;
                        mbarrier_wait(q_ready_addr + (qs_n_2) * 8, _parity_q_ready_1);
                        int q_half_base_1_16 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_16 = q_ld_col_lane * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                            : "r"((q_half_base_1_16 + (q_col_bytes_2_16 ^ (q_half_base_1_16 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_17 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_17 = (16 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[4]), "=r"(q_frag[5]), "=r"(q_frag[6]), "=r"(q_frag[7])
                            : "r"((q_half_base_1_17 + (q_col_bytes_2_17 ^ (q_half_base_1_17 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_18 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_18 = (32 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[8]), "=r"(q_frag[9]), "=r"(q_frag[10]), "=r"(q_frag[11])
                            : "r"((q_half_base_1_18 + (q_col_bytes_2_18 ^ (q_half_base_1_18 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_19 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_19 = (48 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[12]), "=r"(q_frag[13]), "=r"(q_frag[14]), "=r"(q_frag[15])
                            : "r"((q_half_base_1_19 + (q_col_bytes_2_19 ^ (q_half_base_1_19 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_20 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_20 = q_ld_col_lane * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[16]), "=r"(q_frag[17]), "=r"(q_frag[18]), "=r"(q_frag[19])
                            : "r"((q_half_base_1_20 + (q_col_bytes_2_20 ^ (q_half_base_1_20 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_21 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_21 = (16 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[20]), "=r"(q_frag[21]), "=r"(q_frag[22]), "=r"(q_frag[23])
                            : "r"((q_half_base_1_21 + (q_col_bytes_2_21 ^ (q_half_base_1_21 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_22 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_22 = (32 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[24]), "=r"(q_frag[25]), "=r"(q_frag[26]), "=r"(q_frag[27])
                            : "r"((q_half_base_1_22 + (q_col_bytes_2_22 ^ (q_half_base_1_22 >> 7 & 7) << 4)))
                            : "memory");
                        int q_half_base_1_23 = q_smem_addr + (unsigned int)(qs_n_2 * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                        int q_col_bytes_2_23 = (48 + q_ld_col_lane) * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(q_frag[28]), "=r"(q_frag[29]), "=r"(q_frag[30]), "=r"(q_frag[31])
                            : "r"((q_half_base_1_23 + (q_col_bytes_2_23 ^ (q_half_base_1_23 >> 7 & 7) << 4)))
                            : "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(q_empty_addr);
                        }
                        if (nx[5] > 0) {
                            int mbase_n_0_3 = (ti_3 + 1 & 1) * 144;
                            int e0_n_2 = meta_smem[mbase_n_0_3 + 76 + cwg * 34] & 65535;
                            int own_n_2 = gbase + (e0_n_2 >> 3);
                            nx[8] = own_n_2;
                            nx[9] = e0_n_2 >> 2 & 1;
                            #pragma unroll 1
                            for (int p_5 = prev + 1; p_5 < nx[8]; p_5++) {
                                int plap_4 = 0;
                                plap_4 += p_5 / 3;
                                int _parity_k_full_7 = (int)((unsigned int)plap_4 % 2);
                                mbarrier_wait(k_full_addr + (p_5 % 3) * 8, _parity_k_full_7);
                                if (elect_sync()) {
                                    mbarrier_arrive(k_empty_addr + (p_5 % 3) * 8);
                                }
                                int _parity_v_full_6 = (int)((unsigned int)plap_4 % 2);
                                mbarrier_wait(v_full_addr + (p_5 % 3) * 8, _parity_v_full_6);
                                if (elect_sync()) {
                                    mbarrier_arrive(v_empty_addr + (p_5 % 3) * 8);
                                }
                            }
                            int kown_2 = 0;
                            kown_2 += own_n_2;
                            int klown_2 = 0;
                            klown_2 += own_n_2 / 3;
                            int _parity_k_full_8 = (int)((unsigned int)klown_2 % 2);
                            mbarrier_wait(k_full_addr + (kown_2 % 3) * 8, _parity_k_full_8);
                        }
                    }
                }
                int stage_n_0 = nx[8] % 3;
                int real_n_1 = 0;
                if (nx[7] != 0) {
                    if (nx[5] > 0) {
                        real_n_1 = 1;
                    }
                }
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                if (real_n_1 != 0) {
                    uint64_t _wgmma_desc_120 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_n_0 * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_120 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_120);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(wgmma_b_4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(wgmma_b_4 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(wgmma_b_4 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(wgmma_b_4 + 6)
                        : "memory");
                    uint64_t _wgmma_desc_121 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_n_0 * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b_1_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_121 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_121);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(wgmma_b_1_3)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(wgmma_b_1_3 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(wgmma_b_1_3 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(wgmma_b_1_3 + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                } else {
                    uint64_t _wgmma_desc_122 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_n_0 * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_12 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_122 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_122);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_12)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                }
                int store_wg = 1;
                int partner_idle = n_other == 0;
                int self_idle = n_own == 0;
                int skip_merge = ((partner_idle != 0) ? 1 : ((self_idle != 0) ? 1 : 0));
                if (self_idle != 0) {
                    store_wg = 0;
                }
                int do_merge = ((is_split != 0) ? 1 : 0);
                if (skip_merge != 0) {
                    do_merge = 0;
                }
                float a0 = 1.0f;
                float a1 = 1.0f;
                float b0 = 0.0f;
                float b1 = 0.0f;
                if (do_merge != 0) {
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (cwg == 1) {
                        merge_o[tid_wg] = d_o[0];
                        merge_o[128 + tid_wg] = d_o[1];
                        merge_o[256 + tid_wg] = d_o[2];
                        merge_o[384 + tid_wg] = d_o[3];
                        merge_o[512 + tid_wg] = d_o[4];
                        merge_o[640 + tid_wg] = d_o[5];
                        merge_o[768 + tid_wg] = d_o[6];
                        merge_o[896 + tid_wg] = d_o[7];
                        merge_o[1024 + tid_wg] = d_o[8];
                        merge_o[1152 + tid_wg] = d_o[9];
                        merge_o[1280 + tid_wg] = d_o[10];
                        merge_o[1408 + tid_wg] = d_o[11];
                        merge_o[1536 + tid_wg] = d_o[12];
                        merge_o[1664 + tid_wg] = d_o[13];
                        merge_o[1792 + tid_wg] = d_o[14];
                        merge_o[1920 + tid_wg] = d_o[15];
                        merge_o[2048 + tid_wg] = d_o[16];
                        merge_o[2176 + tid_wg] = d_o[17];
                        merge_o[2304 + tid_wg] = d_o[18];
                        merge_o[2432 + tid_wg] = d_o[19];
                        merge_o[2560 + tid_wg] = d_o[20];
                        merge_o[2688 + tid_wg] = d_o[21];
                        merge_o[2816 + tid_wg] = d_o[22];
                        merge_o[2944 + tid_wg] = d_o[23];
                        merge_o[3072 + tid_wg] = d_o[24];
                        merge_o[3200 + tid_wg] = d_o[25];
                        merge_o[3328 + tid_wg] = d_o[26];
                        merge_o[3456 + tid_wg] = d_o[27];
                        merge_o[3584 + tid_wg] = d_o[28];
                        merge_o[3712 + tid_wg] = d_o[29];
                        merge_o[3840 + tid_wg] = d_o[30];
                        merge_o[3968 + tid_wg] = d_o[31];
                        if ((lane & 3) == 0) {
                            merge_ml[quad * 4] = row_max0;
                            merge_ml[quad * 4 + 1] = row_max1;
                            merge_ml[quad * 4 + 2] = row_sum0;
                            merge_ml[quad * 4 + 3] = row_sum1;
                        }
                        store_wg = 0;
                    }
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (cwg == 0) {
                        float pm0 = merge_ml[quad * 4];
                        float pm1 = merge_ml[quad * 4 + 1];
                        float pl0 = merge_ml[quad * 4 + 2];
                        float pl1 = merge_ml[quad * 4 + 3];
                        float mm0 = fmaxf(row_max0, pm0);
                        float mm1 = fmaxf(row_max1, pm1);
                        float _exp2_123 = approx_exp2(row_max0 - mm0);
                        a0 = ((row_max0 == -CUDART_INF_F) ? 0.0f : _exp2_123);
                        float _exp2_124 = approx_exp2(row_max1 - mm1);
                        a1 = ((row_max1 == -CUDART_INF_F) ? 0.0f : _exp2_124);
                        float _exp2_125 = approx_exp2(pm0 - mm0);
                        b0 = ((pm0 == -CUDART_INF_F) ? 0.0f : _exp2_125);
                        float _exp2_126 = approx_exp2(pm1 - mm1);
                        b1 = ((pm1 == -CUDART_INF_F) ? 0.0f : _exp2_126);
                        float po_31_0 = merge_o[tid_wg];
                        d_o[0] = __fmaf_rn(d_o[0], a0, po_31_0 * b0);
                        float po_31_1 = merge_o[128 + tid_wg];
                        d_o[1] = __fmaf_rn(d_o[1], a0, po_31_1 * b0);
                        float po_31_2 = merge_o[256 + tid_wg];
                        d_o[2] = __fmaf_rn(d_o[2], a1, po_31_2 * b1);
                        float po_31_3 = merge_o[384 + tid_wg];
                        d_o[3] = __fmaf_rn(d_o[3], a1, po_31_3 * b1);
                        float po_31_4 = merge_o[512 + tid_wg];
                        d_o[4] = __fmaf_rn(d_o[4], a0, po_31_4 * b0);
                        float po_31_5 = merge_o[640 + tid_wg];
                        d_o[5] = __fmaf_rn(d_o[5], a0, po_31_5 * b0);
                        float po_31_6 = merge_o[768 + tid_wg];
                        d_o[6] = __fmaf_rn(d_o[6], a1, po_31_6 * b1);
                        float po_31_7 = merge_o[896 + tid_wg];
                        d_o[7] = __fmaf_rn(d_o[7], a1, po_31_7 * b1);
                        float po_31_8 = merge_o[1024 + tid_wg];
                        d_o[8] = __fmaf_rn(d_o[8], a0, po_31_8 * b0);
                        float po_31_9 = merge_o[1152 + tid_wg];
                        d_o[9] = __fmaf_rn(d_o[9], a0, po_31_9 * b0);
                        float po_31_10 = merge_o[1280 + tid_wg];
                        d_o[10] = __fmaf_rn(d_o[10], a1, po_31_10 * b1);
                        float po_31_11 = merge_o[1408 + tid_wg];
                        d_o[11] = __fmaf_rn(d_o[11], a1, po_31_11 * b1);
                        float po_31_12 = merge_o[1536 + tid_wg];
                        d_o[12] = __fmaf_rn(d_o[12], a0, po_31_12 * b0);
                        float po_31_13 = merge_o[1664 + tid_wg];
                        d_o[13] = __fmaf_rn(d_o[13], a0, po_31_13 * b0);
                        float po_31_14 = merge_o[1792 + tid_wg];
                        d_o[14] = __fmaf_rn(d_o[14], a1, po_31_14 * b1);
                        float po_31_15 = merge_o[1920 + tid_wg];
                        d_o[15] = __fmaf_rn(d_o[15], a1, po_31_15 * b1);
                        float po_31_16 = merge_o[2048 + tid_wg];
                        d_o[16] = __fmaf_rn(d_o[16], a0, po_31_16 * b0);
                        float po_31_17 = merge_o[2176 + tid_wg];
                        d_o[17] = __fmaf_rn(d_o[17], a0, po_31_17 * b0);
                        float po_31_18 = merge_o[2304 + tid_wg];
                        d_o[18] = __fmaf_rn(d_o[18], a1, po_31_18 * b1);
                        float po_31_19 = merge_o[2432 + tid_wg];
                        d_o[19] = __fmaf_rn(d_o[19], a1, po_31_19 * b1);
                        float po_31_20 = merge_o[2560 + tid_wg];
                        d_o[20] = __fmaf_rn(d_o[20], a0, po_31_20 * b0);
                        float po_31_21 = merge_o[2688 + tid_wg];
                        d_o[21] = __fmaf_rn(d_o[21], a0, po_31_21 * b0);
                        float po_31_22 = merge_o[2816 + tid_wg];
                        d_o[22] = __fmaf_rn(d_o[22], a1, po_31_22 * b1);
                        float po_31_23 = merge_o[2944 + tid_wg];
                        d_o[23] = __fmaf_rn(d_o[23], a1, po_31_23 * b1);
                        float po_31_24 = merge_o[3072 + tid_wg];
                        d_o[24] = __fmaf_rn(d_o[24], a0, po_31_24 * b0);
                        float po_31_25 = merge_o[3200 + tid_wg];
                        d_o[25] = __fmaf_rn(d_o[25], a0, po_31_25 * b0);
                        float po_31_26 = merge_o[3328 + tid_wg];
                        d_o[26] = __fmaf_rn(d_o[26], a1, po_31_26 * b1);
                        float po_31_27 = merge_o[3456 + tid_wg];
                        d_o[27] = __fmaf_rn(d_o[27], a1, po_31_27 * b1);
                        float po_31_28 = merge_o[3584 + tid_wg];
                        d_o[28] = __fmaf_rn(d_o[28], a0, po_31_28 * b0);
                        float po_31_29 = merge_o[3712 + tid_wg];
                        d_o[29] = __fmaf_rn(d_o[29], a0, po_31_29 * b0);
                        float po_31_30 = merge_o[3840 + tid_wg];
                        d_o[30] = __fmaf_rn(d_o[30], a1, po_31_30 * b1);
                        float po_31_31 = merge_o[3968 + tid_wg];
                        d_o[31] = __fmaf_rn(d_o[31], a1, po_31_31 * b1);
                        row_sum0 = __fmaf_rn(row_sum0, a0, pl0 * b0);
                        row_sum1 = __fmaf_rn(row_sum1, a1, pl1 * b1);
                    }
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (cwg == 1) {
                        merge_o[tid_wg] = d_o[32];
                        merge_o[128 + tid_wg] = d_o[33];
                        merge_o[256 + tid_wg] = d_o[34];
                        merge_o[384 + tid_wg] = d_o[35];
                        merge_o[512 + tid_wg] = d_o[36];
                        merge_o[640 + tid_wg] = d_o[37];
                        merge_o[768 + tid_wg] = d_o[38];
                        merge_o[896 + tid_wg] = d_o[39];
                        merge_o[1024 + tid_wg] = d_o[40];
                        merge_o[1152 + tid_wg] = d_o[41];
                        merge_o[1280 + tid_wg] = d_o[42];
                        merge_o[1408 + tid_wg] = d_o[43];
                        merge_o[1536 + tid_wg] = d_o[44];
                        merge_o[1664 + tid_wg] = d_o[45];
                        merge_o[1792 + tid_wg] = d_o[46];
                        merge_o[1920 + tid_wg] = d_o[47];
                        merge_o[2048 + tid_wg] = d_o[48];
                        merge_o[2176 + tid_wg] = d_o[49];
                        merge_o[2304 + tid_wg] = d_o[50];
                        merge_o[2432 + tid_wg] = d_o[51];
                        merge_o[2560 + tid_wg] = d_o[52];
                        merge_o[2688 + tid_wg] = d_o[53];
                        merge_o[2816 + tid_wg] = d_o[54];
                        merge_o[2944 + tid_wg] = d_o[55];
                        merge_o[3072 + tid_wg] = d_o[56];
                        merge_o[3200 + tid_wg] = d_o[57];
                        merge_o[3328 + tid_wg] = d_o[58];
                        merge_o[3456 + tid_wg] = d_o[59];
                        merge_o[3584 + tid_wg] = d_o[60];
                        merge_o[3712 + tid_wg] = d_o[61];
                        merge_o[3840 + tid_wg] = d_o[62];
                        merge_o[3968 + tid_wg] = d_o[63];
                    }
                    asm volatile("barrier.sync 3, 256;" ::: "memory");
                    if (cwg == 0) {
                        float po_h_31_0 = merge_o[tid_wg];
                        d_o[32] = __fmaf_rn(d_o[32], a0, po_h_31_0 * b0);
                        float po_h_31_1 = merge_o[128 + tid_wg];
                        d_o[33] = __fmaf_rn(d_o[33], a0, po_h_31_1 * b0);
                        float po_h_31_2 = merge_o[256 + tid_wg];
                        d_o[34] = __fmaf_rn(d_o[34], a1, po_h_31_2 * b1);
                        float po_h_31_3 = merge_o[384 + tid_wg];
                        d_o[35] = __fmaf_rn(d_o[35], a1, po_h_31_3 * b1);
                        float po_h_31_4 = merge_o[512 + tid_wg];
                        d_o[36] = __fmaf_rn(d_o[36], a0, po_h_31_4 * b0);
                        float po_h_31_5 = merge_o[640 + tid_wg];
                        d_o[37] = __fmaf_rn(d_o[37], a0, po_h_31_5 * b0);
                        float po_h_31_6 = merge_o[768 + tid_wg];
                        d_o[38] = __fmaf_rn(d_o[38], a1, po_h_31_6 * b1);
                        float po_h_31_7 = merge_o[896 + tid_wg];
                        d_o[39] = __fmaf_rn(d_o[39], a1, po_h_31_7 * b1);
                        float po_h_31_8 = merge_o[1024 + tid_wg];
                        d_o[40] = __fmaf_rn(d_o[40], a0, po_h_31_8 * b0);
                        float po_h_31_9 = merge_o[1152 + tid_wg];
                        d_o[41] = __fmaf_rn(d_o[41], a0, po_h_31_9 * b0);
                        float po_h_31_10 = merge_o[1280 + tid_wg];
                        d_o[42] = __fmaf_rn(d_o[42], a1, po_h_31_10 * b1);
                        float po_h_31_11 = merge_o[1408 + tid_wg];
                        d_o[43] = __fmaf_rn(d_o[43], a1, po_h_31_11 * b1);
                        float po_h_31_12 = merge_o[1536 + tid_wg];
                        d_o[44] = __fmaf_rn(d_o[44], a0, po_h_31_12 * b0);
                        float po_h_31_13 = merge_o[1664 + tid_wg];
                        d_o[45] = __fmaf_rn(d_o[45], a0, po_h_31_13 * b0);
                        float po_h_31_14 = merge_o[1792 + tid_wg];
                        d_o[46] = __fmaf_rn(d_o[46], a1, po_h_31_14 * b1);
                        float po_h_31_15 = merge_o[1920 + tid_wg];
                        d_o[47] = __fmaf_rn(d_o[47], a1, po_h_31_15 * b1);
                        float po_h_31_16 = merge_o[2048 + tid_wg];
                        d_o[48] = __fmaf_rn(d_o[48], a0, po_h_31_16 * b0);
                        float po_h_31_17 = merge_o[2176 + tid_wg];
                        d_o[49] = __fmaf_rn(d_o[49], a0, po_h_31_17 * b0);
                        float po_h_31_18 = merge_o[2304 + tid_wg];
                        d_o[50] = __fmaf_rn(d_o[50], a1, po_h_31_18 * b1);
                        float po_h_31_19 = merge_o[2432 + tid_wg];
                        d_o[51] = __fmaf_rn(d_o[51], a1, po_h_31_19 * b1);
                        float po_h_31_20 = merge_o[2560 + tid_wg];
                        d_o[52] = __fmaf_rn(d_o[52], a0, po_h_31_20 * b0);
                        float po_h_31_21 = merge_o[2688 + tid_wg];
                        d_o[53] = __fmaf_rn(d_o[53], a0, po_h_31_21 * b0);
                        float po_h_31_22 = merge_o[2816 + tid_wg];
                        d_o[54] = __fmaf_rn(d_o[54], a1, po_h_31_22 * b1);
                        float po_h_31_23 = merge_o[2944 + tid_wg];
                        d_o[55] = __fmaf_rn(d_o[55], a1, po_h_31_23 * b1);
                        float po_h_31_24 = merge_o[3072 + tid_wg];
                        d_o[56] = __fmaf_rn(d_o[56], a0, po_h_31_24 * b0);
                        float po_h_31_25 = merge_o[3200 + tid_wg];
                        d_o[57] = __fmaf_rn(d_o[57], a0, po_h_31_25 * b0);
                        float po_h_31_26 = merge_o[3328 + tid_wg];
                        d_o[58] = __fmaf_rn(d_o[58], a1, po_h_31_26 * b1);
                        float po_h_31_27 = merge_o[3456 + tid_wg];
                        d_o[59] = __fmaf_rn(d_o[59], a1, po_h_31_27 * b1);
                        float po_h_31_28 = merge_o[3584 + tid_wg];
                        d_o[60] = __fmaf_rn(d_o[60], a0, po_h_31_28 * b0);
                        float po_h_31_29 = merge_o[3712 + tid_wg];
                        d_o[61] = __fmaf_rn(d_o[61], a0, po_h_31_29 * b0);
                        float po_h_31_30 = merge_o[3840 + tid_wg];
                        d_o[62] = __fmaf_rn(d_o[62], a1, po_h_31_30 * b1);
                        float po_h_31_31 = merge_o[3968 + tid_wg];
                        d_o[63] = __fmaf_rn(d_o[63], a1, po_h_31_31 * b1);
                    }
                }
                if (store_wg != 0) {
                    float _rcp_0 = approx_rcp(row_sum0);
                    float _rcp_1 = approx_rcp(row_sum1);
                    d_o[0] = d_o[0] * _rcp_0;
                    d_o[1] = d_o[1] * _rcp_0;
                    d_o[4] = d_o[4] * _rcp_0;
                    d_o[5] = d_o[5] * _rcp_0;
                    d_o[8] = d_o[8] * _rcp_0;
                    d_o[9] = d_o[9] * _rcp_0;
                    d_o[12] = d_o[12] * _rcp_0;
                    d_o[13] = d_o[13] * _rcp_0;
                    d_o[16] = d_o[16] * _rcp_0;
                    d_o[17] = d_o[17] * _rcp_0;
                    d_o[20] = d_o[20] * _rcp_0;
                    d_o[21] = d_o[21] * _rcp_0;
                    d_o[24] = d_o[24] * _rcp_0;
                    d_o[25] = d_o[25] * _rcp_0;
                    d_o[28] = d_o[28] * _rcp_0;
                    d_o[29] = d_o[29] * _rcp_0;
                    d_o[32] = d_o[32] * _rcp_0;
                    d_o[33] = d_o[33] * _rcp_0;
                    d_o[36] = d_o[36] * _rcp_0;
                    d_o[37] = d_o[37] * _rcp_0;
                    d_o[40] = d_o[40] * _rcp_0;
                    d_o[41] = d_o[41] * _rcp_0;
                    d_o[44] = d_o[44] * _rcp_0;
                    d_o[45] = d_o[45] * _rcp_0;
                    d_o[48] = d_o[48] * _rcp_0;
                    d_o[49] = d_o[49] * _rcp_0;
                    d_o[52] = d_o[52] * _rcp_0;
                    d_o[53] = d_o[53] * _rcp_0;
                    d_o[56] = d_o[56] * _rcp_0;
                    d_o[57] = d_o[57] * _rcp_0;
                    d_o[60] = d_o[60] * _rcp_0;
                    d_o[61] = d_o[61] * _rcp_0;
                    d_o[2] = d_o[2] * _rcp_1;
                    d_o[3] = d_o[3] * _rcp_1;
                    d_o[6] = d_o[6] * _rcp_1;
                    d_o[7] = d_o[7] * _rcp_1;
                    d_o[10] = d_o[10] * _rcp_1;
                    d_o[11] = d_o[11] * _rcp_1;
                    d_o[14] = d_o[14] * _rcp_1;
                    d_o[15] = d_o[15] * _rcp_1;
                    d_o[18] = d_o[18] * _rcp_1;
                    d_o[19] = d_o[19] * _rcp_1;
                    d_o[22] = d_o[22] * _rcp_1;
                    d_o[23] = d_o[23] * _rcp_1;
                    d_o[26] = d_o[26] * _rcp_1;
                    d_o[27] = d_o[27] * _rcp_1;
                    d_o[30] = d_o[30] * _rcp_1;
                    d_o[31] = d_o[31] * _rcp_1;
                    d_o[34] = d_o[34] * _rcp_1;
                    d_o[35] = d_o[35] * _rcp_1;
                    d_o[38] = d_o[38] * _rcp_1;
                    d_o[39] = d_o[39] * _rcp_1;
                    d_o[42] = d_o[42] * _rcp_1;
                    d_o[43] = d_o[43] * _rcp_1;
                    d_o[46] = d_o[46] * _rcp_1;
                    d_o[47] = d_o[47] * _rcp_1;
                    d_o[50] = d_o[50] * _rcp_1;
                    d_o[51] = d_o[51] * _rcp_1;
                    d_o[54] = d_o[54] * _rcp_1;
                    d_o[55] = d_o[55] * _rcp_1;
                    d_o[58] = d_o[58] * _rcp_1;
                    d_o[59] = d_o[59] * _rcp_1;
                    d_o[62] = d_o[62] * _rcp_1;
                    d_o[63] = d_o[63] * _rcp_1;
                    int qj = lane & 3;
                    int qj1 = qj & 1;
                    int qj2 = qj & 2;
                    unsigned int o_vec[4];
                    unsigned int o_tmp[4];
                    int o_row_base = ((head_m + cwg * (1 - is_split)) * seqlen_q + my_qb * 64) * 128;
                    int m_local_r_1_0 = m0_local;
                    __nv_bfloat162 _bit_cast_src_127 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
                    __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
                    __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
                    __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
                    unsigned int _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_131;
                    unsigned int _shfl_xor_sync_132 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_132;
                    unsigned int _shfl_xor_sync_133 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_133;
                    unsigned int _shfl_xor_sync_134 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_134;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_135;
                    unsigned int _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_136;
                    unsigned int _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_137;
                    unsigned int _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_138;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_0 = o_row_base + m_local_r_1_0 * 128 + qj * 8;
                    reinterpret_cast<int4*>(O + o_off_3_0)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_139 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_139)[0];
                    __nv_bfloat162 _bit_cast_src_140 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_140)[0];
                    __nv_bfloat162 _bit_cast_src_141 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_141)[0];
                    __nv_bfloat162 _bit_cast_src_142 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_142)[0];
                    unsigned int _shfl_xor_sync_143 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_143;
                    unsigned int _shfl_xor_sync_144 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_144;
                    unsigned int _shfl_xor_sync_145 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_145;
                    unsigned int _shfl_xor_sync_146 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_146;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_147 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_147;
                    unsigned int _shfl_xor_sync_148 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_148;
                    unsigned int _shfl_xor_sync_149 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_149;
                    unsigned int _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_150;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_1 = o_row_base + m_local_r_1_0 * 128 + (4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_1)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_151 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_151)[0];
                    __nv_bfloat162 _bit_cast_src_152 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_152)[0];
                    __nv_bfloat162 _bit_cast_src_153 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_153)[0];
                    __nv_bfloat162 _bit_cast_src_154 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_154)[0];
                    unsigned int _shfl_xor_sync_155 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_155;
                    unsigned int _shfl_xor_sync_156 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_156;
                    unsigned int _shfl_xor_sync_157 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_157;
                    unsigned int _shfl_xor_sync_158 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_158;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_159;
                    unsigned int _shfl_xor_sync_160 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_160;
                    unsigned int _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_161;
                    unsigned int _shfl_xor_sync_162 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_162;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_2 = o_row_base + m_local_r_1_0 * 128 + (8 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_2)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_163 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_163)[0];
                    __nv_bfloat162 _bit_cast_src_164 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_164)[0];
                    __nv_bfloat162 _bit_cast_src_165 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_165)[0];
                    __nv_bfloat162 _bit_cast_src_166 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_166)[0];
                    unsigned int _shfl_xor_sync_167 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_167;
                    unsigned int _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_168;
                    unsigned int _shfl_xor_sync_169 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_169;
                    unsigned int _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_170;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_171 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_171;
                    unsigned int _shfl_xor_sync_172 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_172;
                    unsigned int _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_173;
                    unsigned int _shfl_xor_sync_174 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_174;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_3 = o_row_base + m_local_r_1_0 * 128 + (12 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_3)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    int m_local_r_1_1 = m1_local;
                    __nv_bfloat162 _bit_cast_src_175 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_175)[0];
                    __nv_bfloat162 _bit_cast_src_176 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_176)[0];
                    __nv_bfloat162 _bit_cast_src_177 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_177)[0];
                    __nv_bfloat162 _bit_cast_src_178 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_178)[0];
                    unsigned int _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_179;
                    unsigned int _shfl_xor_sync_180 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_180;
                    unsigned int _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_181;
                    unsigned int _shfl_xor_sync_182 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_182;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_183;
                    unsigned int _shfl_xor_sync_184 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_184;
                    unsigned int _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_185;
                    unsigned int _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_186;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_4 = o_row_base + m_local_r_1_1 * 128 + qj * 8;
                    reinterpret_cast<int4*>(O + o_off_3_4)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_187 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_187)[0];
                    __nv_bfloat162 _bit_cast_src_188 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_188)[0];
                    __nv_bfloat162 _bit_cast_src_189 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_189)[0];
                    __nv_bfloat162 _bit_cast_src_190 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_190)[0];
                    unsigned int _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_191;
                    unsigned int _shfl_xor_sync_192 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_192;
                    unsigned int _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_193;
                    unsigned int _shfl_xor_sync_194 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_194;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_195;
                    unsigned int _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_196;
                    unsigned int _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_197;
                    unsigned int _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_198;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_5 = o_row_base + m_local_r_1_1 * 128 + (4 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_5)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_199 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_199)[0];
                    __nv_bfloat162 _bit_cast_src_200 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_200)[0];
                    __nv_bfloat162 _bit_cast_src_201 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_201)[0];
                    __nv_bfloat162 _bit_cast_src_202 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_202)[0];
                    unsigned int _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_203;
                    unsigned int _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_204;
                    unsigned int _shfl_xor_sync_205 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_205;
                    unsigned int _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_206;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_207 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_207;
                    unsigned int _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_208;
                    unsigned int _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_209;
                    unsigned int _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_210;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_6 = o_row_base + m_local_r_1_1 * 128 + (8 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_6)[0] = reinterpret_cast<int4*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_211 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
                    o_vec[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_211)[0];
                    __nv_bfloat162 _bit_cast_src_212 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
                    o_vec[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_212)[0];
                    __nv_bfloat162 _bit_cast_src_213 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
                    o_vec[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_213)[0];
                    __nv_bfloat162 _bit_cast_src_214 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
                    o_vec[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_214)[0];
                    unsigned int _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 1);
                    o_tmp[0] = _shfl_xor_sync_215;
                    unsigned int _shfl_xor_sync_216 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 1);
                    o_tmp[1] = _shfl_xor_sync_216;
                    unsigned int _shfl_xor_sync_217 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 1);
                    o_tmp[2] = _shfl_xor_sync_217;
                    unsigned int _shfl_xor_sync_218 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 1);
                    o_tmp[3] = _shfl_xor_sync_218;
                    o_vec[0] = ((qj1 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj1 == 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj1 != 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj1 == 0) ? o_tmp[3] : o_vec[3]);
                    unsigned int _shfl_xor_sync_219 = __shfl_xor_sync(0xFFFFFFFF, o_vec[2], 2);
                    o_tmp[0] = _shfl_xor_sync_219;
                    unsigned int _shfl_xor_sync_220 = __shfl_xor_sync(0xFFFFFFFF, o_vec[3], 2);
                    o_tmp[1] = _shfl_xor_sync_220;
                    unsigned int _shfl_xor_sync_221 = __shfl_xor_sync(0xFFFFFFFF, o_vec[0], 2);
                    o_tmp[2] = _shfl_xor_sync_221;
                    unsigned int _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, o_vec[1], 2);
                    o_tmp[3] = _shfl_xor_sync_222;
                    o_vec[0] = ((qj2 != 0) ? o_tmp[0] : o_vec[0]);
                    o_vec[1] = ((qj2 != 0) ? o_tmp[1] : o_vec[1]);
                    o_vec[2] = ((qj2 == 0) ? o_tmp[2] : o_vec[2]);
                    o_vec[3] = ((qj2 == 0) ? o_tmp[3] : o_vec[3]);
                    int o_off_3_7 = o_row_base + m_local_r_1_1 * 128 + (12 + qj) * 8;
                    reinterpret_cast<int4*>(O + o_off_3_7)[0] = reinterpret_cast<int4*>(o_vec)[0];
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_3 % 2)) * 8);
                }
                if (nx[10] != 0) {
                    if (nx[7] == 0) {
                        break;
                    }
                }
            }
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
