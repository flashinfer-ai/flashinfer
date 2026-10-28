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
#define SMEM_MERGE_O_OFF 17408
#define SMEM_MERGE_O_STAGE_BYTES 16384
#define SMEM_MERGE_O_STRIDE 16384
#define SMEM_MERGE_ML_OFF 231552
#define SMEM_MERGE_ML_STAGE_BYTES 512
#define SMEM_MERGE_ML_STRIDE 512
#define SMEM_SMEM_V7_OFF 148480
#define SMEM_SMEM_V7_STAGE_BYTES 16384
#define SMEM_SMEM_V7_STRIDE 16384
#define SMEM_SMEM_V8_OFF 181248
#define SMEM_SMEM_V8_STAGE_BYTES 16384
#define SMEM_SMEM_V8_STRIDE 16384
#define SMEM_SMEM_V9_OFF 214016
#define SMEM_SMEM_V9_STAGE_BYTES 16384
#define SMEM_SMEM_V9_STRIDE 16384
#define SMEM_TOTAL 232064
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
kernel_cake_vsa_sm90_4dda6062d35c4cb79e02(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, __nv_bfloat16* __restrict__ O, int* __restrict__ meta, const CakeParamArray<int16_t, 1728> hdr, int tile_stride, int seqlen_q, int seqlen_k, float scale_log2, int* __restrict__ dbg, unsigned long long* __restrict__ tl)
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
    float* merge_o = reinterpret_cast<float*>(smem_raw + 17408);
    const int merge_o_addr = smem + 17408;
    float* merge_ml = reinterpret_cast<float*>(smem_raw + 231552);
    const int merge_ml_addr = smem + 231552;
    float* smem_v7 = reinterpret_cast<float*>(smem_raw + 148480);
    const int smem_v7_addr = smem + 148480;
    float* smem_v8 = reinterpret_cast<float*>(smem_raw + 181248);
    const int smem_v8_addr = smem + 181248;
    float* smem_v9 = reinterpret_cast<float*>(smem_raw + 214016);
    const int smem_v9_addr = smem + 214016;
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
                #pragma unroll 1
                for (int ti = 1; ti < nt_st; ti++) {
                    int slot_0 = ti & 1;
                    int mrow_1 = (row0 + ti) * 144;
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
                        if (elect_sync()) {
                            if (ti > 0) {
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
                    meta_smem[slot_0 * 144 + st_t] = w0_2;
                    meta_smem[slot_0 * 144 + st_t + 64] = w1_3;
                    if (st_t + 128 < 144) {
                        meta_smem[slot_0 * 144 + st_t + 128] = w2_4;
                    }
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)ti % 2)) * 8);
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
                        int hblk_a = hdr[hb + 5];
                        int hblk_b = hdr[hb + 6];
                        int kv_row_a = kv_base_h + hblk_a * 64;
                        int kv_row_b = kv_base_h + hblk_b * 64;
                        int has_b = 1;
                        if (hblk_b < 0) {
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
                        int hblk_a_1 = hdr[hb + 5 + 2];
                        int hblk_b_1 = hdr[hb + 6 + 2];
                        int kv_row_a_1 = kv_base_h + hblk_a_1 * 64;
                        int kv_row_b_1 = kv_base_h + hblk_b_1 * 64;
                        int has_b_1 = 1;
                        if (hblk_b_1 < 0) {
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
                        int hblk_a_2 = hdr[hb + 5 + 4];
                        int hblk_b_2 = hdr[hb + 6 + 4];
                        int kv_row_a_2 = kv_base_h + hblk_a_2 * 64;
                        int kv_row_b_2 = kv_base_h + hblk_b_2 * 64;
                        int has_b_2 = 1;
                        if (hblk_b_2 < 0) {
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
                    mbarrier_wait(meta_full_addr, 0);
                    int nt_k = meta_smem[7];
                    #pragma unroll 1
                    for (int ti_1 = 0; ti_1 < nt_k; ti_1++) {
                        int slot_k = ti_1 & 1;
                        int mb_k = slot_k * 144;
                        if (ti_1 > 0) {
                            int _stage_meta_full = (int)((unsigned int)ti_1 % 2);
                            int _parity_meta_full = (int)((unsigned int)ti_1 / 2 % 2);
                            mbarrier_wait(meta_full_addr + (_stage_meta_full) * 8, _parity_meta_full);
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
                        int hvblk_a = hdr[hb + 5];
                        int hvblk_b = hdr[hb + 6];
                        int v_row_a = kv_base_hv + hvblk_a * 64;
                        int v_row_b = kv_base_hv + hvblk_b * 64;
                        int v_has_b = 1;
                        if (hvblk_b < 0) {
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
                        int hvblk_a_1 = hdr[hb + 5 + 2];
                        int hvblk_b_1 = hdr[hb + 6 + 2];
                        int v_row_a_1 = kv_base_hv + hvblk_a_1 * 64;
                        int v_row_b_1 = kv_base_hv + hvblk_b_1 * 64;
                        int v_has_b_1 = 1;
                        if (hvblk_b_1 < 0) {
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
                        int hvblk_a_2 = hdr[hb + 5 + 4];
                        int hvblk_b_2 = hdr[hb + 6 + 4];
                        int v_row_a_2 = kv_base_hv + hvblk_a_2 * 64;
                        int v_row_b_2 = kv_base_hv + hvblk_b_2 * 64;
                        int v_has_b_2 = 1;
                        if (hvblk_b_2 < 0) {
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
                    mbarrier_wait(meta_full_addr, 0);
                    int nt_v = meta_smem[7];
                    #pragma unroll 1
                    for (int ti_2 = 0; ti_2 < nt_v; ti_2++) {
                        int slot_v = ti_2 & 1;
                        int mb_v = slot_v * 144;
                        if (ti_2 > 0) {
                            int _stage_meta_full_1 = (int)((unsigned int)ti_2 % 2);
                            int _parity_meta_full_1 = (int)((unsigned int)ti_2 / 2 % 2);
                            mbarrier_wait(meta_full_addr + (_stage_meta_full_1) * 8, _parity_meta_full_1);
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
            smem_v7[tid_c] = 0.0f;
            smem_v7[256 + tid_c] = 0.0f;
            smem_v7[512 + tid_c] = 0.0f;
            smem_v7[768 + tid_c] = 0.0f;
            smem_v7[1024 + tid_c] = 0.0f;
            smem_v7[1280 + tid_c] = 0.0f;
            smem_v7[1536 + tid_c] = 0.0f;
            smem_v7[1792 + tid_c] = 0.0f;
            smem_v7[2048 + tid_c] = 0.0f;
            smem_v7[2304 + tid_c] = 0.0f;
            smem_v7[2560 + tid_c] = 0.0f;
            smem_v7[2816 + tid_c] = 0.0f;
            smem_v7[3072 + tid_c] = 0.0f;
            smem_v7[3328 + tid_c] = 0.0f;
            smem_v7[3584 + tid_c] = 0.0f;
            smem_v7[3840 + tid_c] = 0.0f;
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
            mbarrier_wait(meta_full_addr, 0);
            int nt_m = __shfl_sync(0xFFFFFFFF, meta_smem[7], 0);
            #pragma unroll 1
            for (int ti_3 = 0; ti_3 < nt_m; ti_3++) {
                int slot_m = ti_3 & 1;
                int mbase = slot_m * 144;
                if (ti_3 > 0) {
                    int _stage_meta_full_2 = (int)((unsigned int)ti_3 % 2);
                    int _parity_meta_full_2 = (int)((unsigned int)ti_3 / 2 % 2);
                    mbarrier_wait(meta_full_addr + (_stage_meta_full_2) * 8, _parity_meta_full_2);
                }
                int head_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase], 0);
                int qb0_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 1], 0);
                int qb1_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 2], 0);
                int n_seq_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 4], 0);
                int n_own = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + cwg], 0);
                int n_other = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + 1 - cwg], 0);
                int mode_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 3], 0);
                int is_split = mode_m == 1;
                int my_qb = qb0_m;
                int qs = ((is_split != 0) ? 0 : cwg);
                int own_base_w = mbase + 76 + cwg * 34;
                if (is_split == 0) {
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
                }
                row_max0 = -CUDART_INF_F;
                row_max1 = -CUDART_INF_F;
                row_sum0 = 0.0f;
                row_sum1 = 0.0f;
                int _parity_q_ready = (int)((unsigned int)ti_3 % 2);
                mbarrier_wait(q_ready_addr + (qs) * 8, _parity_q_ready);
                int q_half_base = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
                    : "r"((q_half_base + (q_col_bytes ^ (q_half_base >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_0 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_1 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[4]), "=r"(q_frag[5]), "=r"(q_frag[6]), "=r"(q_frag[7])
                    : "r"((q_half_base_0 + (q_col_bytes_1 ^ (q_half_base_0 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_2 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_3 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[8]), "=r"(q_frag[9]), "=r"(q_frag[10]), "=r"(q_frag[11])
                    : "r"((q_half_base_2 + (q_col_bytes_3 ^ (q_half_base_2 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_4 = q_smem_addr + (unsigned int)(qs * 16384) + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_5 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[12]), "=r"(q_frag[13]), "=r"(q_frag[14]), "=r"(q_frag[15])
                    : "r"((q_half_base_4 + (q_col_bytes_5 ^ (q_half_base_4 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_6 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_7 = q_ld_col_lane * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[16]), "=r"(q_frag[17]), "=r"(q_frag[18]), "=r"(q_frag[19])
                    : "r"((q_half_base_6 + (q_col_bytes_7 ^ (q_half_base_6 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_8 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_9 = (16 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[20]), "=r"(q_frag[21]), "=r"(q_frag[22]), "=r"(q_frag[23])
                    : "r"((q_half_base_8 + (q_col_bytes_9 ^ (q_half_base_8 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_10 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_11 = (32 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[24]), "=r"(q_frag[25]), "=r"(q_frag[26]), "=r"(q_frag[27])
                    : "r"((q_half_base_10 + (q_col_bytes_11 ^ (q_half_base_10 >> 7 & 7) << 4)))
                    : "memory");
                int q_half_base_12 = q_smem_addr + (unsigned int)(qs * 16384) + 8192 + (unsigned int)(q_ld_row * 128);
                int q_col_bytes_13 = (48 + q_ld_col_lane) * 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_frag[28]), "=r"(q_frag[29]), "=r"(q_frag[30]), "=r"(q_frag[31])
                    : "r"((q_half_base_12 + (q_col_bytes_13 ^ (q_half_base_12 >> 7 & 7) << 4)))
                    : "memory");
                if (elect_sync()) {
                    mbarrier_arrive(q_empty_addr);
                }
                int cur_pos = 0;
                int cur_has2 = 1;
                if (n_own > 0) {
                    int kpos0 = 0;
                    int klap0 = 0;
                    int w_e = meta_smem[own_base_w];
                    int e0 = w_e & 65535;
                    int pos0 = gbase + (e0 >> 3);
                    int has2_0 = e0 >> 2 & 1;
                    #pragma unroll 1
                    for (int p_1 = prev + 1; p_1 < pos0; p_1++) {
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
                    kpos0 += pos0;
                    klap0 += pos0 / 3;
                    cur_pos = pos0;
                    cur_has2 = has2_0;
                    int stage_0 = pos0 % 3;
                    int _parity_k_full_1 = (int)((unsigned int)klap0 % 2);
                    mbarrier_wait(k_full_addr + (kpos0 % 3) * 8, _parity_k_full_1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_0 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_0 * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
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
                    uint64_t _wgmma_desc_1 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_0 * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
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
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (kpos0 % 3) * 8);
                    }
                    float f_max0 = -CUDART_INF_F;
                    float f_max1 = -CUDART_INF_F;
                    float m0 = -CUDART_INF_F;
                    float m1 = -CUDART_INF_F;
                    if (scale_log2 >= 0.0f) {
                        float _max_0 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])), fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]))), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])), fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]))));
                        float _max_1 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])), fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]))), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])), fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]))));
                        float _max_2 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37])), fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]))), fmaxf(fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53])), fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]))));
                        float _max_3 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39])), fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]))), fmaxf(fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55])), fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]))));
                        m0 = ((has2_0 != 0) ? fmaxf(_max_0, _max_2) : _max_0);
                        m1 = ((has2_0 != 0) ? fmaxf(_max_1, _max_3) : _max_1);
                        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        m0 = fmaxf(m0, _shfl_xor_sync_2);
                        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        m0 = fmaxf(m0, _shfl_xor_sync_3);
                        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        m1 = fmaxf(m1, _shfl_xor_sync_4);
                        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        m1 = fmaxf(m1, _shfl_xor_sync_5);
                    } else {
                        float _min_0 = fminf(fminf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])), fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]))), fminf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])), fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]))));
                        float _min_1 = fminf(fminf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])), fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]))), fminf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])), fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]))));
                        float _min_2 = fminf(fminf(fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37])), fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]))), fminf(fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53])), fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]))));
                        float _min_3 = fminf(fminf(fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39])), fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]))), fminf(fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55])), fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]))));
                        m0 = ((has2_0 != 0) ? fminf(_min_0, _min_2) : _min_0);
                        m1 = ((has2_0 != 0) ? fminf(_min_1, _min_3) : _min_1);
                        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                        m0 = fminf(m0, _shfl_xor_sync_6);
                        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                        m0 = fminf(m0, _shfl_xor_sync_7);
                        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                        m1 = fminf(m1, _shfl_xor_sync_8);
                        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                        m1 = fminf(m1, _shfl_xor_sync_9);
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
                    float _exp2_0 = approx_exp2(__fmaf_rn(d_qk[0], scale_log2, -row_max0));
                    float _exp2_1 = approx_exp2(__fmaf_rn(d_qk[1], scale_log2, -row_max0));
                    d_qk[0] = _exp2_0;
                    d_qk[1] = _exp2_1;
                    sa0 += _exp2_0 + _exp2_1;
                    float _exp2_2 = approx_exp2(__fmaf_rn(d_qk[2], scale_log2, -row_max1));
                    float _exp2_3 = approx_exp2(__fmaf_rn(d_qk[3], scale_log2, -row_max1));
                    d_qk[2] = _exp2_2;
                    d_qk[3] = _exp2_3;
                    sa1 += _exp2_2 + _exp2_3;
                    float _exp2_4 = approx_exp2(__fmaf_rn(d_qk[4], scale_log2, -row_max0));
                    float _exp2_5 = approx_exp2(__fmaf_rn(d_qk[5], scale_log2, -row_max0));
                    d_qk[4] = _exp2_4;
                    d_qk[5] = _exp2_5;
                    sa0 += _exp2_4 + _exp2_5;
                    float _exp2_6 = approx_exp2(__fmaf_rn(d_qk[6], scale_log2, -row_max1));
                    float _exp2_7 = approx_exp2(__fmaf_rn(d_qk[7], scale_log2, -row_max1));
                    d_qk[6] = _exp2_6;
                    d_qk[7] = _exp2_7;
                    sa1 += _exp2_6 + _exp2_7;
                    float _exp2_8 = approx_exp2(__fmaf_rn(d_qk[8], scale_log2, -row_max0));
                    float _exp2_9 = approx_exp2(__fmaf_rn(d_qk[9], scale_log2, -row_max0));
                    d_qk[8] = _exp2_8;
                    d_qk[9] = _exp2_9;
                    sa0 += _exp2_8 + _exp2_9;
                    float _exp2_10 = approx_exp2(__fmaf_rn(d_qk[10], scale_log2, -row_max1));
                    float _exp2_11 = approx_exp2(__fmaf_rn(d_qk[11], scale_log2, -row_max1));
                    d_qk[10] = _exp2_10;
                    d_qk[11] = _exp2_11;
                    sa1 += _exp2_10 + _exp2_11;
                    float _exp2_12 = approx_exp2(__fmaf_rn(d_qk[12], scale_log2, -row_max0));
                    float _exp2_13 = approx_exp2(__fmaf_rn(d_qk[13], scale_log2, -row_max0));
                    d_qk[12] = _exp2_12;
                    d_qk[13] = _exp2_13;
                    sa0 += _exp2_12 + _exp2_13;
                    float _exp2_14 = approx_exp2(__fmaf_rn(d_qk[14], scale_log2, -row_max1));
                    float _exp2_15 = approx_exp2(__fmaf_rn(d_qk[15], scale_log2, -row_max1));
                    d_qk[14] = _exp2_14;
                    d_qk[15] = _exp2_15;
                    sa1 += _exp2_14 + _exp2_15;
                    float _exp2_16 = approx_exp2(__fmaf_rn(d_qk[16], scale_log2, -row_max0));
                    float _exp2_17 = approx_exp2(__fmaf_rn(d_qk[17], scale_log2, -row_max0));
                    d_qk[16] = _exp2_16;
                    d_qk[17] = _exp2_17;
                    sa0 += _exp2_16 + _exp2_17;
                    float _exp2_18 = approx_exp2(__fmaf_rn(d_qk[18], scale_log2, -row_max1));
                    float _exp2_19 = approx_exp2(__fmaf_rn(d_qk[19], scale_log2, -row_max1));
                    d_qk[18] = _exp2_18;
                    d_qk[19] = _exp2_19;
                    sa1 += _exp2_18 + _exp2_19;
                    float _exp2_20 = approx_exp2(__fmaf_rn(d_qk[20], scale_log2, -row_max0));
                    float _exp2_21 = approx_exp2(__fmaf_rn(d_qk[21], scale_log2, -row_max0));
                    d_qk[20] = _exp2_20;
                    d_qk[21] = _exp2_21;
                    sa0 += _exp2_20 + _exp2_21;
                    float _exp2_22 = approx_exp2(__fmaf_rn(d_qk[22], scale_log2, -row_max1));
                    float _exp2_23 = approx_exp2(__fmaf_rn(d_qk[23], scale_log2, -row_max1));
                    d_qk[22] = _exp2_22;
                    d_qk[23] = _exp2_23;
                    sa1 += _exp2_22 + _exp2_23;
                    float _exp2_24 = approx_exp2(__fmaf_rn(d_qk[24], scale_log2, -row_max0));
                    float _exp2_25 = approx_exp2(__fmaf_rn(d_qk[25], scale_log2, -row_max0));
                    d_qk[24] = _exp2_24;
                    d_qk[25] = _exp2_25;
                    sa0 += _exp2_24 + _exp2_25;
                    float _exp2_26 = approx_exp2(__fmaf_rn(d_qk[26], scale_log2, -row_max1));
                    float _exp2_27 = approx_exp2(__fmaf_rn(d_qk[27], scale_log2, -row_max1));
                    d_qk[26] = _exp2_26;
                    d_qk[27] = _exp2_27;
                    sa1 += _exp2_26 + _exp2_27;
                    float _exp2_28 = approx_exp2(__fmaf_rn(d_qk[28], scale_log2, -row_max0));
                    float _exp2_29 = approx_exp2(__fmaf_rn(d_qk[29], scale_log2, -row_max0));
                    d_qk[28] = _exp2_28;
                    d_qk[29] = _exp2_29;
                    sa0 += _exp2_28 + _exp2_29;
                    float _exp2_30 = approx_exp2(__fmaf_rn(d_qk[30], scale_log2, -row_max1));
                    float _exp2_31 = approx_exp2(__fmaf_rn(d_qk[31], scale_log2, -row_max1));
                    d_qk[30] = _exp2_30;
                    d_qk[31] = _exp2_31;
                    sa1 += _exp2_30 + _exp2_31;
                    float _exp2_32 = approx_exp2(__fmaf_rn(d_qk[32], scale_log2, -row_max0));
                    float _exp2_33 = approx_exp2(__fmaf_rn(d_qk[33], scale_log2, -row_max0));
                    d_qk[32] = _exp2_32;
                    d_qk[33] = _exp2_33;
                    sb0 += _exp2_32 + _exp2_33;
                    float _exp2_34 = approx_exp2(__fmaf_rn(d_qk[34], scale_log2, -row_max1));
                    float _exp2_35 = approx_exp2(__fmaf_rn(d_qk[35], scale_log2, -row_max1));
                    d_qk[34] = _exp2_34;
                    d_qk[35] = _exp2_35;
                    sb1 += _exp2_34 + _exp2_35;
                    float _exp2_36 = approx_exp2(__fmaf_rn(d_qk[36], scale_log2, -row_max0));
                    float _exp2_37 = approx_exp2(__fmaf_rn(d_qk[37], scale_log2, -row_max0));
                    d_qk[36] = _exp2_36;
                    d_qk[37] = _exp2_37;
                    sb0 += _exp2_36 + _exp2_37;
                    float _exp2_38 = approx_exp2(__fmaf_rn(d_qk[38], scale_log2, -row_max1));
                    float _exp2_39 = approx_exp2(__fmaf_rn(d_qk[39], scale_log2, -row_max1));
                    d_qk[38] = _exp2_38;
                    d_qk[39] = _exp2_39;
                    sb1 += _exp2_38 + _exp2_39;
                    float _exp2_40 = approx_exp2(__fmaf_rn(d_qk[40], scale_log2, -row_max0));
                    float _exp2_41 = approx_exp2(__fmaf_rn(d_qk[41], scale_log2, -row_max0));
                    d_qk[40] = _exp2_40;
                    d_qk[41] = _exp2_41;
                    sb0 += _exp2_40 + _exp2_41;
                    float _exp2_42 = approx_exp2(__fmaf_rn(d_qk[42], scale_log2, -row_max1));
                    float _exp2_43 = approx_exp2(__fmaf_rn(d_qk[43], scale_log2, -row_max1));
                    d_qk[42] = _exp2_42;
                    d_qk[43] = _exp2_43;
                    sb1 += _exp2_42 + _exp2_43;
                    float _exp2_44 = approx_exp2(__fmaf_rn(d_qk[44], scale_log2, -row_max0));
                    float _exp2_45 = approx_exp2(__fmaf_rn(d_qk[45], scale_log2, -row_max0));
                    d_qk[44] = _exp2_44;
                    d_qk[45] = _exp2_45;
                    sb0 += _exp2_44 + _exp2_45;
                    float _exp2_46 = approx_exp2(__fmaf_rn(d_qk[46], scale_log2, -row_max1));
                    float _exp2_47 = approx_exp2(__fmaf_rn(d_qk[47], scale_log2, -row_max1));
                    d_qk[46] = _exp2_46;
                    d_qk[47] = _exp2_47;
                    sb1 += _exp2_46 + _exp2_47;
                    float _exp2_48 = approx_exp2(__fmaf_rn(d_qk[48], scale_log2, -row_max0));
                    float _exp2_49 = approx_exp2(__fmaf_rn(d_qk[49], scale_log2, -row_max0));
                    d_qk[48] = _exp2_48;
                    d_qk[49] = _exp2_49;
                    sb0 += _exp2_48 + _exp2_49;
                    float _exp2_50 = approx_exp2(__fmaf_rn(d_qk[50], scale_log2, -row_max1));
                    float _exp2_51 = approx_exp2(__fmaf_rn(d_qk[51], scale_log2, -row_max1));
                    d_qk[50] = _exp2_50;
                    d_qk[51] = _exp2_51;
                    sb1 += _exp2_50 + _exp2_51;
                    float _exp2_52 = approx_exp2(__fmaf_rn(d_qk[52], scale_log2, -row_max0));
                    float _exp2_53 = approx_exp2(__fmaf_rn(d_qk[53], scale_log2, -row_max0));
                    d_qk[52] = _exp2_52;
                    d_qk[53] = _exp2_53;
                    sb0 += _exp2_52 + _exp2_53;
                    float _exp2_54 = approx_exp2(__fmaf_rn(d_qk[54], scale_log2, -row_max1));
                    float _exp2_55 = approx_exp2(__fmaf_rn(d_qk[55], scale_log2, -row_max1));
                    d_qk[54] = _exp2_54;
                    d_qk[55] = _exp2_55;
                    sb1 += _exp2_54 + _exp2_55;
                    float _exp2_56 = approx_exp2(__fmaf_rn(d_qk[56], scale_log2, -row_max0));
                    float _exp2_57 = approx_exp2(__fmaf_rn(d_qk[57], scale_log2, -row_max0));
                    d_qk[56] = _exp2_56;
                    d_qk[57] = _exp2_57;
                    sb0 += _exp2_56 + _exp2_57;
                    float _exp2_58 = approx_exp2(__fmaf_rn(d_qk[58], scale_log2, -row_max1));
                    float _exp2_59 = approx_exp2(__fmaf_rn(d_qk[59], scale_log2, -row_max1));
                    d_qk[58] = _exp2_58;
                    d_qk[59] = _exp2_59;
                    sb1 += _exp2_58 + _exp2_59;
                    float _exp2_60 = approx_exp2(__fmaf_rn(d_qk[60], scale_log2, -row_max0));
                    float _exp2_61 = approx_exp2(__fmaf_rn(d_qk[61], scale_log2, -row_max0));
                    d_qk[60] = _exp2_60;
                    d_qk[61] = _exp2_61;
                    sb0 += _exp2_60 + _exp2_61;
                    float _exp2_62 = approx_exp2(__fmaf_rn(d_qk[62], scale_log2, -row_max1));
                    float _exp2_63 = approx_exp2(__fmaf_rn(d_qk[63], scale_log2, -row_max1));
                    d_qk[62] = _exp2_62;
                    d_qk[63] = _exp2_63;
                    sb1 += _exp2_62 + _exp2_63;
                    f_sum0 += sa0 + ((has2_0 != 0) ? sb0 : 0.0f);
                    f_sum1 += sa1 + ((has2_0 != 0) ? sb1 : 0.0f);
                    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 2);
                    f_sum0 += _shfl_xor_sync_10;
                    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 1);
                    f_sum0 += _shfl_xor_sync_11;
                    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 2);
                    f_sum1 += _shfl_xor_sync_12;
                    float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 1);
                    f_sum1 += _shfl_xor_sync_13;
                    row_sum0 = f_sum0;
                    row_sum1 = f_sum1;
                    __nv_bfloat162 _bit_cast_src_14 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
                    p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_14)[0];
                    __nv_bfloat162 _bit_cast_src_15 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
                    p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_15)[0];
                    __nv_bfloat162 _bit_cast_src_16 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
                    p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_16)[0];
                    __nv_bfloat162 _bit_cast_src_17 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
                    p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_17)[0];
                    __nv_bfloat162 _bit_cast_src_18 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
                    p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_18)[0];
                    __nv_bfloat162 _bit_cast_src_19 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
                    p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_19)[0];
                    __nv_bfloat162 _bit_cast_src_20 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
                    p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_20)[0];
                    __nv_bfloat162 _bit_cast_src_21 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
                    p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_21)[0];
                    __nv_bfloat162 _bit_cast_src_22 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
                    p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_22)[0];
                    __nv_bfloat162 _bit_cast_src_23 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
                    p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_23)[0];
                    __nv_bfloat162 _bit_cast_src_24 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
                    p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_24)[0];
                    __nv_bfloat162 _bit_cast_src_25 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
                    p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_25)[0];
                    __nv_bfloat162 _bit_cast_src_26 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
                    p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_26)[0];
                    __nv_bfloat162 _bit_cast_src_27 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
                    p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_27)[0];
                    __nv_bfloat162 _bit_cast_src_28 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
                    p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_28)[0];
                    __nv_bfloat162 _bit_cast_src_29 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
                    p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_29)[0];
                    __nv_bfloat162 _bit_cast_src_30 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                    p_bf16[16] = reinterpret_cast<unsigned int*>(&_bit_cast_src_30)[0];
                    __nv_bfloat162 _bit_cast_src_31 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                    p_bf16[17] = reinterpret_cast<unsigned int*>(&_bit_cast_src_31)[0];
                    __nv_bfloat162 _bit_cast_src_32 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                    p_bf16[18] = reinterpret_cast<unsigned int*>(&_bit_cast_src_32)[0];
                    __nv_bfloat162 _bit_cast_src_33 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                    p_bf16[19] = reinterpret_cast<unsigned int*>(&_bit_cast_src_33)[0];
                    __nv_bfloat162 _bit_cast_src_34 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                    p_bf16[20] = reinterpret_cast<unsigned int*>(&_bit_cast_src_34)[0];
                    __nv_bfloat162 _bit_cast_src_35 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                    p_bf16[21] = reinterpret_cast<unsigned int*>(&_bit_cast_src_35)[0];
                    __nv_bfloat162 _bit_cast_src_36 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                    p_bf16[22] = reinterpret_cast<unsigned int*>(&_bit_cast_src_36)[0];
                    __nv_bfloat162 _bit_cast_src_37 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                    p_bf16[23] = reinterpret_cast<unsigned int*>(&_bit_cast_src_37)[0];
                    __nv_bfloat162 _bit_cast_src_38 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                    p_bf16[24] = reinterpret_cast<unsigned int*>(&_bit_cast_src_38)[0];
                    __nv_bfloat162 _bit_cast_src_39 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                    p_bf16[25] = reinterpret_cast<unsigned int*>(&_bit_cast_src_39)[0];
                    __nv_bfloat162 _bit_cast_src_40 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                    p_bf16[26] = reinterpret_cast<unsigned int*>(&_bit_cast_src_40)[0];
                    __nv_bfloat162 _bit_cast_src_41 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                    p_bf16[27] = reinterpret_cast<unsigned int*>(&_bit_cast_src_41)[0];
                    __nv_bfloat162 _bit_cast_src_42 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                    p_bf16[28] = reinterpret_cast<unsigned int*>(&_bit_cast_src_42)[0];
                    __nv_bfloat162 _bit_cast_src_43 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                    p_bf16[29] = reinterpret_cast<unsigned int*>(&_bit_cast_src_43)[0];
                    __nv_bfloat162 _bit_cast_src_44 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                    p_bf16[30] = reinterpret_cast<unsigned int*>(&_bit_cast_src_44)[0];
                    __nv_bfloat162 _bit_cast_src_45 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                    p_bf16[31] = reinterpret_cast<unsigned int*>(&_bit_cast_src_45)[0];
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
                    int n_loop = n_own - 1;
                    #pragma unroll 1
                    for (int n = 0; n < n_loop; n++) {
                        int cur_stage = cur_pos % 3;
                        int w_e_0 = meta_smem[own_base_w + (n + 1 >> 1)];
                        int en = w_e_0 >> (n + 1 & 1) * 16 & 65535;
                        int nxt_pos = gbase + (en >> 3);
                        int nxt_has2 = en >> 2 & 1;
                        #pragma unroll 1
                        for (int p_2 = cur_pos + 1; p_2 < nxt_pos; p_2++) {
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
                        int knext = 0;
                        knext += nxt_pos;
                        int knlap = 0;
                        knlap += nxt_pos / 3;
                        int vcur = 0;
                        vcur += cur_pos;
                        int vclap = 0;
                        vclap += cur_pos / 3;
                        int nxt_stage = nxt_pos % 3;
                        int _parity_k_full_3 = (int)((unsigned int)knlap % 2);
                        mbarrier_wait(k_full_addr + (knext % 3) * 8, _parity_k_full_3);
                        int _parity_v_full_2 = (int)((unsigned int)vclap % 2);
                        mbarrier_wait(v_full_addr + (vcur % 3) * 8, _parity_v_full_2);
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_46 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_46 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_46);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(_wgmma_b_2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(_wgmma_b_2 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(_wgmma_b_2 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(_wgmma_b_2 + 6)
                            : "memory");
                        uint64_t _wgmma_desc_47 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_47 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_47);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[16]), "r"(q_frag[(16) + 1]), "r"(q_frag[(16) + 2]), "r"(q_frag[(16) + 3]), "l"(_wgmma_b_3)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[20]), "r"(q_frag[(20) + 1]), "r"(q_frag[(20) + 2]), "r"(q_frag[(20) + 3]), "l"(_wgmma_b_3 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[24]), "r"(q_frag[(24) + 1]), "r"(q_frag[(24) + 2]), "r"(q_frag[(24) + 3]), "l"(_wgmma_b_3 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[28]), "r"(q_frag[(28) + 1]), "r"(q_frag[(28) + 2]), "r"(q_frag[(28) + 3]), "l"(_wgmma_b_3 + 6)
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
                        uint64_t _wgmma_desc_48 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_48 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_48);
                        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_4), "r"(n + (1 - is_split))
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_4 + 128)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_4 + 256)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_4 + 384)
                            : "memory");
                        uint64_t _wgmma_desc_49 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(cur_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_49 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_49);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(_wgmma_b_5)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(_wgmma_b_5 + 128)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(_wgmma_b_5 + 256)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(_wgmma_b_5 + 384)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (knext % 3) * 8);
                        }
                        float new_max0 = -CUDART_INF_F;
                        float new_max1 = -CUDART_INF_F;
                        float m0_1 = -CUDART_INF_F;
                        float m1_2 = -CUDART_INF_F;
                        if (scale_log2 >= 0.0f) {
                            float _max_4 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5])), fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]))), fmaxf(fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21])), fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]))));
                            float _max_5 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7])), fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]))), fmaxf(fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23])), fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]))));
                            float _max_6 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37])), fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]))), fmaxf(fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53])), fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]))));
                            float _max_7 = fmaxf(fmaxf(fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39])), fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]))), fmaxf(fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55])), fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]))));
                            m0_1 = ((nxt_has2 != 0) ? fmaxf(_max_4, _max_6) : _max_4);
                            m1_2 = ((nxt_has2 != 0) ? fmaxf(_max_5, _max_7) : _max_5);
                            float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
                            m0_1 = fmaxf(m0_1, _shfl_xor_sync_50);
                            float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
                            m0_1 = fmaxf(m0_1, _shfl_xor_sync_51);
                            float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 2);
                            m1_2 = fmaxf(m1_2, _shfl_xor_sync_52);
                            float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 1);
                            m1_2 = fmaxf(m1_2, _shfl_xor_sync_53);
                        } else {
                            float _min_4 = fminf(fminf(fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5])), fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]))), fminf(fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21])), fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]))));
                            float _min_5 = fminf(fminf(fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7])), fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]))), fminf(fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23])), fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]))));
                            float _min_6 = fminf(fminf(fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37])), fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]))), fminf(fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53])), fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]))));
                            float _min_7 = fminf(fminf(fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39])), fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]))), fminf(fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55])), fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]))));
                            m0_1 = ((nxt_has2 != 0) ? fminf(_min_4, _min_6) : _min_4);
                            m1_2 = ((nxt_has2 != 0) ? fminf(_min_5, _min_7) : _min_5);
                            float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 2);
                            m0_1 = fminf(m0_1, _shfl_xor_sync_54);
                            float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, m0_1, 1);
                            m0_1 = fminf(m0_1, _shfl_xor_sync_55);
                            float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 2);
                            m1_2 = fminf(m1_2, _shfl_xor_sync_56);
                            float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, m1_2, 1);
                            m1_2 = fminf(m1_2, _shfl_xor_sync_57);
                        }
                        new_max0 = m0_1;
                        new_max1 = m1_2;
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
                        float sa0_3 = 0.0f;
                        float sa1_4 = 0.0f;
                        float sb0_5 = 0.0f;
                        float sb1_6 = 0.0f;
                        float _exp2_66 = approx_exp2(__fmaf_rn(d_qk[0], scale_log2, -merged_max0));
                        float _exp2_67 = approx_exp2(__fmaf_rn(d_qk[1], scale_log2, -merged_max0));
                        d_qk[0] = _exp2_66;
                        d_qk[1] = _exp2_67;
                        sa0_3 += _exp2_66 + _exp2_67;
                        float _exp2_68 = approx_exp2(__fmaf_rn(d_qk[2], scale_log2, -merged_max1));
                        float _exp2_69 = approx_exp2(__fmaf_rn(d_qk[3], scale_log2, -merged_max1));
                        d_qk[2] = _exp2_68;
                        d_qk[3] = _exp2_69;
                        sa1_4 += _exp2_68 + _exp2_69;
                        float _exp2_70 = approx_exp2(__fmaf_rn(d_qk[4], scale_log2, -merged_max0));
                        float _exp2_71 = approx_exp2(__fmaf_rn(d_qk[5], scale_log2, -merged_max0));
                        d_qk[4] = _exp2_70;
                        d_qk[5] = _exp2_71;
                        sa0_3 += _exp2_70 + _exp2_71;
                        float _exp2_72 = approx_exp2(__fmaf_rn(d_qk[6], scale_log2, -merged_max1));
                        float _exp2_73 = approx_exp2(__fmaf_rn(d_qk[7], scale_log2, -merged_max1));
                        d_qk[6] = _exp2_72;
                        d_qk[7] = _exp2_73;
                        sa1_4 += _exp2_72 + _exp2_73;
                        float _exp2_74 = approx_exp2(__fmaf_rn(d_qk[8], scale_log2, -merged_max0));
                        float _exp2_75 = approx_exp2(__fmaf_rn(d_qk[9], scale_log2, -merged_max0));
                        d_qk[8] = _exp2_74;
                        d_qk[9] = _exp2_75;
                        sa0_3 += _exp2_74 + _exp2_75;
                        float _exp2_76 = approx_exp2(__fmaf_rn(d_qk[10], scale_log2, -merged_max1));
                        float _exp2_77 = approx_exp2(__fmaf_rn(d_qk[11], scale_log2, -merged_max1));
                        d_qk[10] = _exp2_76;
                        d_qk[11] = _exp2_77;
                        sa1_4 += _exp2_76 + _exp2_77;
                        float _exp2_78 = approx_exp2(__fmaf_rn(d_qk[12], scale_log2, -merged_max0));
                        float _exp2_79 = approx_exp2(__fmaf_rn(d_qk[13], scale_log2, -merged_max0));
                        d_qk[12] = _exp2_78;
                        d_qk[13] = _exp2_79;
                        sa0_3 += _exp2_78 + _exp2_79;
                        float _exp2_80 = approx_exp2(__fmaf_rn(d_qk[14], scale_log2, -merged_max1));
                        float _exp2_81 = approx_exp2(__fmaf_rn(d_qk[15], scale_log2, -merged_max1));
                        d_qk[14] = _exp2_80;
                        d_qk[15] = _exp2_81;
                        sa1_4 += _exp2_80 + _exp2_81;
                        float _exp2_82 = approx_exp2(__fmaf_rn(d_qk[16], scale_log2, -merged_max0));
                        float _exp2_83 = approx_exp2(__fmaf_rn(d_qk[17], scale_log2, -merged_max0));
                        d_qk[16] = _exp2_82;
                        d_qk[17] = _exp2_83;
                        sa0_3 += _exp2_82 + _exp2_83;
                        float _exp2_84 = approx_exp2(__fmaf_rn(d_qk[18], scale_log2, -merged_max1));
                        float _exp2_85 = approx_exp2(__fmaf_rn(d_qk[19], scale_log2, -merged_max1));
                        d_qk[18] = _exp2_84;
                        d_qk[19] = _exp2_85;
                        sa1_4 += _exp2_84 + _exp2_85;
                        float _exp2_86 = approx_exp2(__fmaf_rn(d_qk[20], scale_log2, -merged_max0));
                        float _exp2_87 = approx_exp2(__fmaf_rn(d_qk[21], scale_log2, -merged_max0));
                        d_qk[20] = _exp2_86;
                        d_qk[21] = _exp2_87;
                        sa0_3 += _exp2_86 + _exp2_87;
                        float _exp2_88 = approx_exp2(__fmaf_rn(d_qk[22], scale_log2, -merged_max1));
                        float _exp2_89 = approx_exp2(__fmaf_rn(d_qk[23], scale_log2, -merged_max1));
                        d_qk[22] = _exp2_88;
                        d_qk[23] = _exp2_89;
                        sa1_4 += _exp2_88 + _exp2_89;
                        float _exp2_90 = approx_exp2(__fmaf_rn(d_qk[24], scale_log2, -merged_max0));
                        float _exp2_91 = approx_exp2(__fmaf_rn(d_qk[25], scale_log2, -merged_max0));
                        d_qk[24] = _exp2_90;
                        d_qk[25] = _exp2_91;
                        sa0_3 += _exp2_90 + _exp2_91;
                        float _exp2_92 = approx_exp2(__fmaf_rn(d_qk[26], scale_log2, -merged_max1));
                        float _exp2_93 = approx_exp2(__fmaf_rn(d_qk[27], scale_log2, -merged_max1));
                        d_qk[26] = _exp2_92;
                        d_qk[27] = _exp2_93;
                        sa1_4 += _exp2_92 + _exp2_93;
                        float _exp2_94 = approx_exp2(__fmaf_rn(d_qk[28], scale_log2, -merged_max0));
                        float _exp2_95 = approx_exp2(__fmaf_rn(d_qk[29], scale_log2, -merged_max0));
                        d_qk[28] = _exp2_94;
                        d_qk[29] = _exp2_95;
                        sa0_3 += _exp2_94 + _exp2_95;
                        float _exp2_96 = approx_exp2(__fmaf_rn(d_qk[30], scale_log2, -merged_max1));
                        float _exp2_97 = approx_exp2(__fmaf_rn(d_qk[31], scale_log2, -merged_max1));
                        d_qk[30] = _exp2_96;
                        d_qk[31] = _exp2_97;
                        sa1_4 += _exp2_96 + _exp2_97;
                        float _exp2_98 = approx_exp2(__fmaf_rn(d_qk[32], scale_log2, -merged_max0));
                        float _exp2_99 = approx_exp2(__fmaf_rn(d_qk[33], scale_log2, -merged_max0));
                        d_qk[32] = _exp2_98;
                        d_qk[33] = _exp2_99;
                        sb0_5 += _exp2_98 + _exp2_99;
                        float _exp2_100 = approx_exp2(__fmaf_rn(d_qk[34], scale_log2, -merged_max1));
                        float _exp2_101 = approx_exp2(__fmaf_rn(d_qk[35], scale_log2, -merged_max1));
                        d_qk[34] = _exp2_100;
                        d_qk[35] = _exp2_101;
                        sb1_6 += _exp2_100 + _exp2_101;
                        float _exp2_102 = approx_exp2(__fmaf_rn(d_qk[36], scale_log2, -merged_max0));
                        float _exp2_103 = approx_exp2(__fmaf_rn(d_qk[37], scale_log2, -merged_max0));
                        d_qk[36] = _exp2_102;
                        d_qk[37] = _exp2_103;
                        sb0_5 += _exp2_102 + _exp2_103;
                        float _exp2_104 = approx_exp2(__fmaf_rn(d_qk[38], scale_log2, -merged_max1));
                        float _exp2_105 = approx_exp2(__fmaf_rn(d_qk[39], scale_log2, -merged_max1));
                        d_qk[38] = _exp2_104;
                        d_qk[39] = _exp2_105;
                        sb1_6 += _exp2_104 + _exp2_105;
                        float _exp2_106 = approx_exp2(__fmaf_rn(d_qk[40], scale_log2, -merged_max0));
                        float _exp2_107 = approx_exp2(__fmaf_rn(d_qk[41], scale_log2, -merged_max0));
                        d_qk[40] = _exp2_106;
                        d_qk[41] = _exp2_107;
                        sb0_5 += _exp2_106 + _exp2_107;
                        float _exp2_108 = approx_exp2(__fmaf_rn(d_qk[42], scale_log2, -merged_max1));
                        float _exp2_109 = approx_exp2(__fmaf_rn(d_qk[43], scale_log2, -merged_max1));
                        d_qk[42] = _exp2_108;
                        d_qk[43] = _exp2_109;
                        sb1_6 += _exp2_108 + _exp2_109;
                        float _exp2_110 = approx_exp2(__fmaf_rn(d_qk[44], scale_log2, -merged_max0));
                        float _exp2_111 = approx_exp2(__fmaf_rn(d_qk[45], scale_log2, -merged_max0));
                        d_qk[44] = _exp2_110;
                        d_qk[45] = _exp2_111;
                        sb0_5 += _exp2_110 + _exp2_111;
                        float _exp2_112 = approx_exp2(__fmaf_rn(d_qk[46], scale_log2, -merged_max1));
                        float _exp2_113 = approx_exp2(__fmaf_rn(d_qk[47], scale_log2, -merged_max1));
                        d_qk[46] = _exp2_112;
                        d_qk[47] = _exp2_113;
                        sb1_6 += _exp2_112 + _exp2_113;
                        float _exp2_114 = approx_exp2(__fmaf_rn(d_qk[48], scale_log2, -merged_max0));
                        float _exp2_115 = approx_exp2(__fmaf_rn(d_qk[49], scale_log2, -merged_max0));
                        d_qk[48] = _exp2_114;
                        d_qk[49] = _exp2_115;
                        sb0_5 += _exp2_114 + _exp2_115;
                        float _exp2_116 = approx_exp2(__fmaf_rn(d_qk[50], scale_log2, -merged_max1));
                        float _exp2_117 = approx_exp2(__fmaf_rn(d_qk[51], scale_log2, -merged_max1));
                        d_qk[50] = _exp2_116;
                        d_qk[51] = _exp2_117;
                        sb1_6 += _exp2_116 + _exp2_117;
                        float _exp2_118 = approx_exp2(__fmaf_rn(d_qk[52], scale_log2, -merged_max0));
                        float _exp2_119 = approx_exp2(__fmaf_rn(d_qk[53], scale_log2, -merged_max0));
                        d_qk[52] = _exp2_118;
                        d_qk[53] = _exp2_119;
                        sb0_5 += _exp2_118 + _exp2_119;
                        float _exp2_120 = approx_exp2(__fmaf_rn(d_qk[54], scale_log2, -merged_max1));
                        float _exp2_121 = approx_exp2(__fmaf_rn(d_qk[55], scale_log2, -merged_max1));
                        d_qk[54] = _exp2_120;
                        d_qk[55] = _exp2_121;
                        sb1_6 += _exp2_120 + _exp2_121;
                        float _exp2_122 = approx_exp2(__fmaf_rn(d_qk[56], scale_log2, -merged_max0));
                        float _exp2_123 = approx_exp2(__fmaf_rn(d_qk[57], scale_log2, -merged_max0));
                        d_qk[56] = _exp2_122;
                        d_qk[57] = _exp2_123;
                        sb0_5 += _exp2_122 + _exp2_123;
                        float _exp2_124 = approx_exp2(__fmaf_rn(d_qk[58], scale_log2, -merged_max1));
                        float _exp2_125 = approx_exp2(__fmaf_rn(d_qk[59], scale_log2, -merged_max1));
                        d_qk[58] = _exp2_124;
                        d_qk[59] = _exp2_125;
                        sb1_6 += _exp2_124 + _exp2_125;
                        float _exp2_126 = approx_exp2(__fmaf_rn(d_qk[60], scale_log2, -merged_max0));
                        float _exp2_127 = approx_exp2(__fmaf_rn(d_qk[61], scale_log2, -merged_max0));
                        d_qk[60] = _exp2_126;
                        d_qk[61] = _exp2_127;
                        sb0_5 += _exp2_126 + _exp2_127;
                        float _exp2_128 = approx_exp2(__fmaf_rn(d_qk[62], scale_log2, -merged_max1));
                        float _exp2_129 = approx_exp2(__fmaf_rn(d_qk[63], scale_log2, -merged_max1));
                        d_qk[62] = _exp2_128;
                        d_qk[63] = _exp2_129;
                        sb1_6 += _exp2_128 + _exp2_129;
                        new_sum0 += sa0_3 + ((nxt_has2 != 0) ? sb0_5 : 0.0f);
                        new_sum1 += sa1_4 + ((nxt_has2 != 0) ? sb1_6 : 0.0f);
                        float _shfl_xor_sync_58 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 2);
                        new_sum0 += _shfl_xor_sync_58;
                        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 1);
                        new_sum0 += _shfl_xor_sync_59;
                        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 2);
                        new_sum1 += _shfl_xor_sync_60;
                        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 1);
                        new_sum1 += _shfl_xor_sync_61;
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (vcur % 3) * 8);
                        }
                        pend0 = _exp2_64;
                        pend1 = _exp2_65;
                        row_sum0 = __fmaf_rn(row_sum0, _exp2_64, new_sum0);
                        row_sum1 = __fmaf_rn(row_sum1, _exp2_65, new_sum1);
                        __nv_bfloat162 _bit_cast_src_62 = __float22bfloat162_rn(make_float2(d_qk[0], d_qk[1]));
                        p_bf16[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_62)[0];
                        __nv_bfloat162 _bit_cast_src_63 = __float22bfloat162_rn(make_float2(d_qk[2], d_qk[3]));
                        p_bf16[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_63)[0];
                        __nv_bfloat162 _bit_cast_src_64 = __float22bfloat162_rn(make_float2(d_qk[4], d_qk[5]));
                        p_bf16[2] = reinterpret_cast<unsigned int*>(&_bit_cast_src_64)[0];
                        __nv_bfloat162 _bit_cast_src_65 = __float22bfloat162_rn(make_float2(d_qk[6], d_qk[7]));
                        p_bf16[3] = reinterpret_cast<unsigned int*>(&_bit_cast_src_65)[0];
                        __nv_bfloat162 _bit_cast_src_66 = __float22bfloat162_rn(make_float2(d_qk[8], d_qk[9]));
                        p_bf16[4] = reinterpret_cast<unsigned int*>(&_bit_cast_src_66)[0];
                        __nv_bfloat162 _bit_cast_src_67 = __float22bfloat162_rn(make_float2(d_qk[10], d_qk[11]));
                        p_bf16[5] = reinterpret_cast<unsigned int*>(&_bit_cast_src_67)[0];
                        __nv_bfloat162 _bit_cast_src_68 = __float22bfloat162_rn(make_float2(d_qk[12], d_qk[13]));
                        p_bf16[6] = reinterpret_cast<unsigned int*>(&_bit_cast_src_68)[0];
                        __nv_bfloat162 _bit_cast_src_69 = __float22bfloat162_rn(make_float2(d_qk[14], d_qk[15]));
                        p_bf16[7] = reinterpret_cast<unsigned int*>(&_bit_cast_src_69)[0];
                        __nv_bfloat162 _bit_cast_src_70 = __float22bfloat162_rn(make_float2(d_qk[16], d_qk[17]));
                        p_bf16[8] = reinterpret_cast<unsigned int*>(&_bit_cast_src_70)[0];
                        __nv_bfloat162 _bit_cast_src_71 = __float22bfloat162_rn(make_float2(d_qk[18], d_qk[19]));
                        p_bf16[9] = reinterpret_cast<unsigned int*>(&_bit_cast_src_71)[0];
                        __nv_bfloat162 _bit_cast_src_72 = __float22bfloat162_rn(make_float2(d_qk[20], d_qk[21]));
                        p_bf16[10] = reinterpret_cast<unsigned int*>(&_bit_cast_src_72)[0];
                        __nv_bfloat162 _bit_cast_src_73 = __float22bfloat162_rn(make_float2(d_qk[22], d_qk[23]));
                        p_bf16[11] = reinterpret_cast<unsigned int*>(&_bit_cast_src_73)[0];
                        __nv_bfloat162 _bit_cast_src_74 = __float22bfloat162_rn(make_float2(d_qk[24], d_qk[25]));
                        p_bf16[12] = reinterpret_cast<unsigned int*>(&_bit_cast_src_74)[0];
                        __nv_bfloat162 _bit_cast_src_75 = __float22bfloat162_rn(make_float2(d_qk[26], d_qk[27]));
                        p_bf16[13] = reinterpret_cast<unsigned int*>(&_bit_cast_src_75)[0];
                        __nv_bfloat162 _bit_cast_src_76 = __float22bfloat162_rn(make_float2(d_qk[28], d_qk[29]));
                        p_bf16[14] = reinterpret_cast<unsigned int*>(&_bit_cast_src_76)[0];
                        __nv_bfloat162 _bit_cast_src_77 = __float22bfloat162_rn(make_float2(d_qk[30], d_qk[31]));
                        p_bf16[15] = reinterpret_cast<unsigned int*>(&_bit_cast_src_77)[0];
                        __nv_bfloat162 _bit_cast_src_78 = __float22bfloat162_rn(make_float2(d_qk[32], d_qk[33]));
                        p_bf16[16] = reinterpret_cast<unsigned int*>(&_bit_cast_src_78)[0];
                        __nv_bfloat162 _bit_cast_src_79 = __float22bfloat162_rn(make_float2(d_qk[34], d_qk[35]));
                        p_bf16[17] = reinterpret_cast<unsigned int*>(&_bit_cast_src_79)[0];
                        __nv_bfloat162 _bit_cast_src_80 = __float22bfloat162_rn(make_float2(d_qk[36], d_qk[37]));
                        p_bf16[18] = reinterpret_cast<unsigned int*>(&_bit_cast_src_80)[0];
                        __nv_bfloat162 _bit_cast_src_81 = __float22bfloat162_rn(make_float2(d_qk[38], d_qk[39]));
                        p_bf16[19] = reinterpret_cast<unsigned int*>(&_bit_cast_src_81)[0];
                        __nv_bfloat162 _bit_cast_src_82 = __float22bfloat162_rn(make_float2(d_qk[40], d_qk[41]));
                        p_bf16[20] = reinterpret_cast<unsigned int*>(&_bit_cast_src_82)[0];
                        __nv_bfloat162 _bit_cast_src_83 = __float22bfloat162_rn(make_float2(d_qk[42], d_qk[43]));
                        p_bf16[21] = reinterpret_cast<unsigned int*>(&_bit_cast_src_83)[0];
                        __nv_bfloat162 _bit_cast_src_84 = __float22bfloat162_rn(make_float2(d_qk[44], d_qk[45]));
                        p_bf16[22] = reinterpret_cast<unsigned int*>(&_bit_cast_src_84)[0];
                        __nv_bfloat162 _bit_cast_src_85 = __float22bfloat162_rn(make_float2(d_qk[46], d_qk[47]));
                        p_bf16[23] = reinterpret_cast<unsigned int*>(&_bit_cast_src_85)[0];
                        __nv_bfloat162 _bit_cast_src_86 = __float22bfloat162_rn(make_float2(d_qk[48], d_qk[49]));
                        p_bf16[24] = reinterpret_cast<unsigned int*>(&_bit_cast_src_86)[0];
                        __nv_bfloat162 _bit_cast_src_87 = __float22bfloat162_rn(make_float2(d_qk[50], d_qk[51]));
                        p_bf16[25] = reinterpret_cast<unsigned int*>(&_bit_cast_src_87)[0];
                        __nv_bfloat162 _bit_cast_src_88 = __float22bfloat162_rn(make_float2(d_qk[52], d_qk[53]));
                        p_bf16[26] = reinterpret_cast<unsigned int*>(&_bit_cast_src_88)[0];
                        __nv_bfloat162 _bit_cast_src_89 = __float22bfloat162_rn(make_float2(d_qk[54], d_qk[55]));
                        p_bf16[27] = reinterpret_cast<unsigned int*>(&_bit_cast_src_89)[0];
                        __nv_bfloat162 _bit_cast_src_90 = __float22bfloat162_rn(make_float2(d_qk[56], d_qk[57]));
                        p_bf16[28] = reinterpret_cast<unsigned int*>(&_bit_cast_src_90)[0];
                        __nv_bfloat162 _bit_cast_src_91 = __float22bfloat162_rn(make_float2(d_qk[58], d_qk[59]));
                        p_bf16[29] = reinterpret_cast<unsigned int*>(&_bit_cast_src_91)[0];
                        __nv_bfloat162 _bit_cast_src_92 = __float22bfloat162_rn(make_float2(d_qk[60], d_qk[61]));
                        p_bf16[30] = reinterpret_cast<unsigned int*>(&_bit_cast_src_92)[0];
                        __nv_bfloat162 _bit_cast_src_93 = __float22bfloat162_rn(make_float2(d_qk[62], d_qk[63]));
                        p_bf16[31] = reinterpret_cast<unsigned int*>(&_bit_cast_src_93)[0];
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
                    int _parity_v_full_3 = (int)((unsigned int)vllap % 2);
                    mbarrier_wait(v_full_addr + (vlast % 3) * 8, _parity_v_full_3);
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
                    uint64_t _wgmma_desc_94 = (((uint64_t)(((vt_smem_a_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_94 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_94);
                    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[0]), "r"(p_bf16[1]), "r"(p_bf16[2]), "r"(p_bf16[3]), "l"(_wgmma_b_6), "r"(n_own - 1 + (1 - is_split))
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[4]), "r"(p_bf16[(4) + 1]), "r"(p_bf16[(4) + 2]), "r"(p_bf16[(4) + 3]), "l"(_wgmma_b_6 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[8]), "r"(p_bf16[(8) + 1]), "r"(p_bf16[(8) + 2]), "r"(p_bf16[(8) + 3]), "l"(_wgmma_b_6 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[12]), "r"(p_bf16[(12) + 1]), "r"(p_bf16[(12) + 2]), "r"(p_bf16[(12) + 3]), "l"(_wgmma_b_6 + 384)
                        : "memory");
                    uint64_t _wgmma_desc_95 = (((uint64_t)(((vt_smem_b_addr + (unsigned int)(last_stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(512) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_7 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_95 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_95);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[16]), "r"(p_bf16[(16) + 1]), "r"(p_bf16[(16) + 2]), "r"(p_bf16[(16) + 3]), "l"(_wgmma_b_7)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[20]), "r"(p_bf16[(20) + 1]), "r"(p_bf16[(20) + 2]), "r"(p_bf16[(20) + 3]), "l"(_wgmma_b_7 + 128)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[24]), "r"(p_bf16[(24) + 1]), "r"(p_bf16[(24) + 2]), "r"(p_bf16[(24) + 3]), "l"(_wgmma_b_7 + 256)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_bf16[28]), "r"(p_bf16[(28) + 1]), "r"(p_bf16[(28) + 2]), "r"(p_bf16[(28) + 3]), "l"(_wgmma_b_7 + 384)
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
                for (int p_3 = tail_lo; p_3 < tile_end; p_3++) {
                    int tlap = 0;
                    tlap += p_3 / 3;
                    int _parity_k_full_4 = (int)((unsigned int)tlap % 2);
                    mbarrier_wait(k_full_addr + (p_3 % 3) * 8, _parity_k_full_4);
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (p_3 % 3) * 8);
                    }
                    int _parity_v_full_4 = (int)((unsigned int)tlap % 2);
                    mbarrier_wait(v_full_addr + (p_3 % 3) * 8, _parity_v_full_4);
                    if (elect_sync()) {
                        mbarrier_arrive(v_empty_addr + (p_3 % 3) * 8);
                    }
                }
                prev = tile_end - 1;
                gbase = tile_end;
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
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
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
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
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (cwg == 0) {
                        float pm0 = merge_ml[quad * 4];
                        float pm1 = merge_ml[quad * 4 + 1];
                        float pl0 = merge_ml[quad * 4 + 2];
                        float pl1 = merge_ml[quad * 4 + 3];
                        float mm0 = fmaxf(row_max0, pm0);
                        float mm1 = fmaxf(row_max1, pm1);
                        float _exp2_96_1 = approx_exp2(row_max0 - mm0);
                        a0 = ((row_max0 == -CUDART_INF_F) ? 0.0f : _exp2_96_1);
                        float _exp2_97_1 = approx_exp2(row_max1 - mm1);
                        a1 = ((row_max1 == -CUDART_INF_F) ? 0.0f : _exp2_97_1);
                        float _exp2_98_1 = approx_exp2(pm0 - mm0);
                        b0 = ((pm0 == -CUDART_INF_F) ? 0.0f : _exp2_98_1);
                        float _exp2_99_1 = approx_exp2(pm1 - mm1);
                        b1 = ((pm1 == -CUDART_INF_F) ? 0.0f : _exp2_99_1);
                        float po = merge_o[tid_wg];
                        d_o[0] = __fmaf_rn(d_o[0], a0, po * b0);
                        float po_0 = merge_o[128 + tid_wg];
                        d_o[1] = __fmaf_rn(d_o[1], a0, po_0 * b0);
                        float po_1 = merge_o[256 + tid_wg];
                        d_o[2] = __fmaf_rn(d_o[2], a1, po_1 * b1);
                        float po_2 = merge_o[384 + tid_wg];
                        d_o[3] = __fmaf_rn(d_o[3], a1, po_2 * b1);
                        float po_3 = merge_o[512 + tid_wg];
                        d_o[4] = __fmaf_rn(d_o[4], a0, po_3 * b0);
                        float po_4 = merge_o[640 + tid_wg];
                        d_o[5] = __fmaf_rn(d_o[5], a0, po_4 * b0);
                        float po_5 = merge_o[768 + tid_wg];
                        d_o[6] = __fmaf_rn(d_o[6], a1, po_5 * b1);
                        float po_6 = merge_o[896 + tid_wg];
                        d_o[7] = __fmaf_rn(d_o[7], a1, po_6 * b1);
                        float po_7 = merge_o[1024 + tid_wg];
                        d_o[8] = __fmaf_rn(d_o[8], a0, po_7 * b0);
                        float po_8 = merge_o[1152 + tid_wg];
                        d_o[9] = __fmaf_rn(d_o[9], a0, po_8 * b0);
                        float po_9 = merge_o[1280 + tid_wg];
                        d_o[10] = __fmaf_rn(d_o[10], a1, po_9 * b1);
                        float po_10 = merge_o[1408 + tid_wg];
                        d_o[11] = __fmaf_rn(d_o[11], a1, po_10 * b1);
                        float po_11 = merge_o[1536 + tid_wg];
                        d_o[12] = __fmaf_rn(d_o[12], a0, po_11 * b0);
                        float po_12 = merge_o[1664 + tid_wg];
                        d_o[13] = __fmaf_rn(d_o[13], a0, po_12 * b0);
                        float po_13 = merge_o[1792 + tid_wg];
                        d_o[14] = __fmaf_rn(d_o[14], a1, po_13 * b1);
                        float po_14 = merge_o[1920 + tid_wg];
                        d_o[15] = __fmaf_rn(d_o[15], a1, po_14 * b1);
                        float po_15 = merge_o[2048 + tid_wg];
                        d_o[16] = __fmaf_rn(d_o[16], a0, po_15 * b0);
                        float po_16 = merge_o[2176 + tid_wg];
                        d_o[17] = __fmaf_rn(d_o[17], a0, po_16 * b0);
                        float po_17 = merge_o[2304 + tid_wg];
                        d_o[18] = __fmaf_rn(d_o[18], a1, po_17 * b1);
                        float po_18 = merge_o[2432 + tid_wg];
                        d_o[19] = __fmaf_rn(d_o[19], a1, po_18 * b1);
                        float po_19 = merge_o[2560 + tid_wg];
                        d_o[20] = __fmaf_rn(d_o[20], a0, po_19 * b0);
                        float po_20 = merge_o[2688 + tid_wg];
                        d_o[21] = __fmaf_rn(d_o[21], a0, po_20 * b0);
                        float po_21 = merge_o[2816 + tid_wg];
                        d_o[22] = __fmaf_rn(d_o[22], a1, po_21 * b1);
                        float po_22 = merge_o[2944 + tid_wg];
                        d_o[23] = __fmaf_rn(d_o[23], a1, po_22 * b1);
                        float po_23 = merge_o[3072 + tid_wg];
                        d_o[24] = __fmaf_rn(d_o[24], a0, po_23 * b0);
                        float po_24 = merge_o[3200 + tid_wg];
                        d_o[25] = __fmaf_rn(d_o[25], a0, po_24 * b0);
                        float po_25 = merge_o[3328 + tid_wg];
                        d_o[26] = __fmaf_rn(d_o[26], a1, po_25 * b1);
                        float po_26 = merge_o[3456 + tid_wg];
                        d_o[27] = __fmaf_rn(d_o[27], a1, po_26 * b1);
                        float po_27 = merge_o[3584 + tid_wg];
                        d_o[28] = __fmaf_rn(d_o[28], a0, po_27 * b0);
                        float po_28 = merge_o[3712 + tid_wg];
                        d_o[29] = __fmaf_rn(d_o[29], a0, po_28 * b0);
                        float po_29 = merge_o[3840 + tid_wg];
                        d_o[30] = __fmaf_rn(d_o[30], a1, po_29 * b1);
                        float po_30 = merge_o[3968 + tid_wg];
                        d_o[31] = __fmaf_rn(d_o[31], a1, po_30 * b1);
                        row_sum0 = __fmaf_rn(row_sum0, a0, pl0 * b0);
                        row_sum1 = __fmaf_rn(row_sum1, a1, pl1 * b1);
                    }
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
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
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                    if (cwg == 0) {
                        float po_h = merge_o[tid_wg];
                        d_o[32] = __fmaf_rn(d_o[32], a0, po_h * b0);
                        float po_h_0 = merge_o[128 + tid_wg];
                        d_o[33] = __fmaf_rn(d_o[33], a0, po_h_0 * b0);
                        float po_h_1 = merge_o[256 + tid_wg];
                        d_o[34] = __fmaf_rn(d_o[34], a1, po_h_1 * b1);
                        float po_h_2 = merge_o[384 + tid_wg];
                        d_o[35] = __fmaf_rn(d_o[35], a1, po_h_2 * b1);
                        float po_h_3 = merge_o[512 + tid_wg];
                        d_o[36] = __fmaf_rn(d_o[36], a0, po_h_3 * b0);
                        float po_h_4 = merge_o[640 + tid_wg];
                        d_o[37] = __fmaf_rn(d_o[37], a0, po_h_4 * b0);
                        float po_h_5 = merge_o[768 + tid_wg];
                        d_o[38] = __fmaf_rn(d_o[38], a1, po_h_5 * b1);
                        float po_h_6 = merge_o[896 + tid_wg];
                        d_o[39] = __fmaf_rn(d_o[39], a1, po_h_6 * b1);
                        float po_h_7 = merge_o[1024 + tid_wg];
                        d_o[40] = __fmaf_rn(d_o[40], a0, po_h_7 * b0);
                        float po_h_8 = merge_o[1152 + tid_wg];
                        d_o[41] = __fmaf_rn(d_o[41], a0, po_h_8 * b0);
                        float po_h_9 = merge_o[1280 + tid_wg];
                        d_o[42] = __fmaf_rn(d_o[42], a1, po_h_9 * b1);
                        float po_h_10 = merge_o[1408 + tid_wg];
                        d_o[43] = __fmaf_rn(d_o[43], a1, po_h_10 * b1);
                        float po_h_11 = merge_o[1536 + tid_wg];
                        d_o[44] = __fmaf_rn(d_o[44], a0, po_h_11 * b0);
                        float po_h_12 = merge_o[1664 + tid_wg];
                        d_o[45] = __fmaf_rn(d_o[45], a0, po_h_12 * b0);
                        float po_h_13 = merge_o[1792 + tid_wg];
                        d_o[46] = __fmaf_rn(d_o[46], a1, po_h_13 * b1);
                        float po_h_14 = merge_o[1920 + tid_wg];
                        d_o[47] = __fmaf_rn(d_o[47], a1, po_h_14 * b1);
                        float po_h_15 = merge_o[2048 + tid_wg];
                        d_o[48] = __fmaf_rn(d_o[48], a0, po_h_15 * b0);
                        float po_h_16 = merge_o[2176 + tid_wg];
                        d_o[49] = __fmaf_rn(d_o[49], a0, po_h_16 * b0);
                        float po_h_17 = merge_o[2304 + tid_wg];
                        d_o[50] = __fmaf_rn(d_o[50], a1, po_h_17 * b1);
                        float po_h_18 = merge_o[2432 + tid_wg];
                        d_o[51] = __fmaf_rn(d_o[51], a1, po_h_18 * b1);
                        float po_h_19 = merge_o[2560 + tid_wg];
                        d_o[52] = __fmaf_rn(d_o[52], a0, po_h_19 * b0);
                        float po_h_20 = merge_o[2688 + tid_wg];
                        d_o[53] = __fmaf_rn(d_o[53], a0, po_h_20 * b0);
                        float po_h_21 = merge_o[2816 + tid_wg];
                        d_o[54] = __fmaf_rn(d_o[54], a1, po_h_21 * b1);
                        float po_h_22 = merge_o[2944 + tid_wg];
                        d_o[55] = __fmaf_rn(d_o[55], a1, po_h_22 * b1);
                        float po_h_23 = merge_o[3072 + tid_wg];
                        d_o[56] = __fmaf_rn(d_o[56], a0, po_h_23 * b0);
                        float po_h_24 = merge_o[3200 + tid_wg];
                        d_o[57] = __fmaf_rn(d_o[57], a0, po_h_24 * b0);
                        float po_h_25 = merge_o[3328 + tid_wg];
                        d_o[58] = __fmaf_rn(d_o[58], a1, po_h_25 * b1);
                        float po_h_26 = merge_o[3456 + tid_wg];
                        d_o[59] = __fmaf_rn(d_o[59], a1, po_h_26 * b1);
                        float po_h_27 = merge_o[3584 + tid_wg];
                        d_o[60] = __fmaf_rn(d_o[60], a0, po_h_27 * b0);
                        float po_h_28 = merge_o[3712 + tid_wg];
                        d_o[61] = __fmaf_rn(d_o[61], a0, po_h_28 * b0);
                        float po_h_29 = merge_o[3840 + tid_wg];
                        d_o[62] = __fmaf_rn(d_o[62], a1, po_h_29 * b1);
                        float po_h_30 = merge_o[3968 + tid_wg];
                        d_o[63] = __fmaf_rn(d_o[63], a1, po_h_30 * b1);
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
                    int qodd = qj & 1;
                    unsigned int o_pk[2];
                    unsigned int o_vec[2];
                    int o_row_base = ((head_m + cwg * (1 - is_split)) * seqlen_q + my_qb * 64) * 128 + qj * 2 + qodd * 6;
                    int m_local_r = m0_local;
                    int row_off = o_row_base + m_local_r * 128;
                    __nv_bfloat162 _bit_cast_src_100 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_100)[0];
                    __nv_bfloat162 _bit_cast_src_101 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_101)[0];
                    unsigned int send = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
                    reinterpret_cast<int2*>(O + row_off)[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_102 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_102)[0];
                    __nv_bfloat162 _bit_cast_src_103 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_103)[0];
                    unsigned int send_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_0, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_104 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_104)[0];
                    __nv_bfloat162 _bit_cast_src_105 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_105)[0];
                    unsigned int send_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_106 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_106)[0];
                    __nv_bfloat162 _bit_cast_src_107 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_107)[0];
                    unsigned int send_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_2, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_108 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_108)[0];
                    __nv_bfloat162 _bit_cast_src_109 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_109)[0];
                    unsigned int send_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_3, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 64))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_110 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_110)[0];
                    __nv_bfloat162 _bit_cast_src_111 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_111)[0];
                    unsigned int send_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_4, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 80))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_112 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_112)[0];
                    __nv_bfloat162 _bit_cast_src_113 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_113)[0];
                    unsigned int send_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_5, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 96))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_114 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_114)[0];
                    __nv_bfloat162 _bit_cast_src_115 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_115)[0];
                    unsigned int send_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_6, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off + 112))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    int m_local_r_7 = m1_local;
                    int row_off_8 = o_row_base + m_local_r_7 * 128;
                    __nv_bfloat162 _bit_cast_src_116 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_116)[0];
                    __nv_bfloat162 _bit_cast_src_117 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_117)[0];
                    unsigned int send_9 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_9, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_8);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_8 : o_pk[1]);
                    reinterpret_cast<int2*>(O + row_off_8)[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_118 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_118)[0];
                    __nv_bfloat162 _bit_cast_src_119 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_119)[0];
                    unsigned int send_10 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_10, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_9);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_9 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_120 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_120)[0];
                    __nv_bfloat162 _bit_cast_src_121 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_121)[0];
                    unsigned int send_11 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_11, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_10);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_10 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_122 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_122)[0];
                    __nv_bfloat162 _bit_cast_src_123 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_123)[0];
                    unsigned int send_12 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_12, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_11);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_11 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_124 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
                    __nv_bfloat162 _bit_cast_src_125 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
                    unsigned int send_13 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_13, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_12);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_12 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 64))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_126 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_126)[0];
                    __nv_bfloat162 _bit_cast_src_127 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
                    unsigned int send_14 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_14, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_13);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_13 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 80))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
                    __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
                    unsigned int send_15 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_15, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_14);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_14 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 96))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
                    __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
                    unsigned int send_16 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_16, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_15);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_15 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_8 + 112))[0] = reinterpret_cast<int2*>(o_vec)[0];
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_3 % 2)) * 8);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
