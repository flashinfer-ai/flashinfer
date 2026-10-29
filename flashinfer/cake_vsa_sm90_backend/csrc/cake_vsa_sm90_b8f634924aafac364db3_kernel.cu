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
#define NUM_VL_PIPE_STAGES 3
#define NUM_Q_PIPE_STAGES 2
#define NUM_META_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 8192
#define SMEM_Q_SMEM_STRIDE 8192
#define SMEM_K_SMEM_OFF 17408
#define SMEM_K_SMEM_STAGE_BYTES 16384
#define SMEM_K_SMEM_STRIDE 16384
#define SMEM_VL_SMEM_OFF 66560
#define SMEM_VL_SMEM_STAGE_BYTES 16384
#define SMEM_VL_SMEM_STRIDE 16384
#define SMEM_VT_SMEM_OFF 115712
#define SMEM_VT_SMEM_STAGE_BYTES 16384
#define SMEM_VT_SMEM_STRIDE 16384
#define SMEM_QS_SMEM_OFF 181248
#define SMEM_QS_SMEM_STAGE_BYTES 512
#define SMEM_QS_SMEM_STRIDE 512
#define SMEM_KS_SMEM_OFF 181760
#define SMEM_KS_SMEM_STAGE_BYTES 768
#define SMEM_KS_SMEM_STRIDE 768
#define SMEM_META_SMEM_OFF 182528
#define SMEM_META_SMEM_STAGE_BYTES 2176
#define SMEM_META_SMEM_STRIDE 2176
#define SMEM_VS_SMEM_OFF 182528
#define SMEM_VS_SMEM_STAGE_BYTES 2176
#define SMEM_VS_SMEM_STRIDE 2176
#define SMEM_TQ_SMEM_OFF 185216
#define SMEM_TQ_SMEM_STAGE_BYTES 16
#define SMEM_TQ_SMEM_STRIDE 16
#define SMEM_MERGE_O_OFF 164864
#define SMEM_MERGE_O_STAGE_BYTES 16384
#define SMEM_MERGE_O_STRIDE 16384
#define SMEM_MERGE_ML_OFF 184704
#define SMEM_MERGE_ML_STAGE_BYTES 512
#define SMEM_MERGE_ML_STRIDE 512
#define SMEM_SMEM_V11_OFF 115712
#define SMEM_SMEM_V11_STAGE_BYTES 16384
#define SMEM_SMEM_V11_STRIDE 16384
#define SMEM_SMEM_V12_OFF 132096
#define SMEM_SMEM_V12_STAGE_BYTES 16384
#define SMEM_SMEM_V12_STRIDE 16384
#define SMEM_SMEM_V13_OFF 148480
#define SMEM_SMEM_V13_STAGE_BYTES 16384
#define SMEM_SMEM_V13_STRIDE 16384
#define SMEM_TOTAL 185344
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
kernel_cake_vsa_sm90_b8f634924aafac364db3(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, const __grid_constant__ CUtensorMap Qs, const __grid_constant__ CUtensorMap Ks, float* __restrict__ Vs, int k_scale_groups, int* __restrict__ meta, const CakeParamArray<int16_t, 1728> hdr, int tile_stride, int seqlen_q, int seqlen_k, float scale_log2, int* __restrict__ dbg, unsigned long long* __restrict__ tl)
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
    #define v_land_full_addr (mbar_base + 120)
    #define v_land_empty_addr (mbar_base + 144)
    #define meta_full_addr (mbar_base + 168)
    #define meta_empty_addr (mbar_base + 184)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* q_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int k_smem_addr = smem + 17408;
    uint8_t* vl_smem = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int vl_smem_addr = smem + 66560;
    uint8_t* vt_smem = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int vt_smem_addr = smem + 115712;
    float* qs_smem = reinterpret_cast<float*>(smem_raw + 181248);
    const int qs_smem_addr = smem + 181248;
    float* ks_smem = reinterpret_cast<float*>(smem_raw + 181760);
    const int ks_smem_addr = smem + 181760;
    int* meta_smem = reinterpret_cast<int*>(smem_raw + 182528);
    const int meta_smem_addr = smem + 182528;
    float* vs_smem = reinterpret_cast<float*>(smem_raw + 182528);
    const int vs_smem_addr = smem + 182528;
    int* tq_smem = reinterpret_cast<int*>(smem_raw + 185216);
    const int tq_smem_addr = smem + 185216;
    float* merge_o = reinterpret_cast<float*>(smem_raw + 164864);
    const int merge_o_addr = smem + 164864;
    float* merge_ml = reinterpret_cast<float*>(smem_raw + 184704);
    const int merge_ml_addr = smem + 184704;
    float* smem_v11 = reinterpret_cast<float*>(smem_raw + 115712);
    const int smem_v11_addr = smem + 115712;
    float* smem_v12 = reinterpret_cast<float*>(smem_raw + 132096);
    const int smem_v12_addr = smem + 132096;
    float* smem_v13 = reinterpret_cast<float*>(smem_raw + 148480);
    const int smem_v13_addr = smem + 148480;
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Qs))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Ks))) : "memory");
        }
    }
    __syncwarp();

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 25 barriers)
    // Mbarriers at smem_raw[0..200)

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
            // v_full: 3 barriers, init_count=32
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
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
            // --- pipeline 'vl_pipe' ---
            // v_land_full: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // v_land_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            // --- pipeline 'meta_pipe' ---
            // meta_full: 2 barriers, init_count=32
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            // meta_empty: 2 barriers, init_count=259
            mbarrier_init(smem + 184, 259);
            mbarrier_init(smem + 192, 259);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: producer ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // producer_main
            int cta = bid;
            int row0 = cta * tile_stride;
            int hb = cta * 12;
            int g_ctas = gridDim.x;
            int q_bound = tile_stride + 256 * g_ctas + 1;
            if (warp != 3) {
                asm volatile("barrier.arrive 1, 384;" ::: "memory");
            }
            if (warp == 2) {
                int st_t = lane;
                int wv[5];
                float vsv[4];
                int h_head_s = hdr[hb];
                int h_kv_s = hdr[hb + 2];
                if (elect_sync()) {
                    int h_qb0 = hdr[hb + 1];
                    int h_qb1 = hdr[hb + 2];
                    int h_mode = hdr[hb + 3];
                    mbarrier_arrive_expect_tx(q_ready_addr, 8448);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(h_head_s * seqlen_q + h_qb0 * 64),
                           "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(qs_smem_addr), "l"((&Qs)), "r"(h_qb0 * 64), "r"(h_head_s),
                           "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                    if (h_mode != 1) {
                        mbarrier_arrive_expect_tx(q_ready_addr + 8, 8448);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(q_smem_addr + 8192), "l"((&Q)), "r"(0), "r"((h_head_s + 1) * seqlen_q + h_qb0 * 64),
                               "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(qs_smem_addr + 256), "l"((&Qs)), "r"(h_qb0 * 64), "r"(h_head_s + 1),
                               "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
                int nt_st_ld = meta[row0 * 144 + 7];
                int slot = 0;
                int mrow = row0 * 144;
                wv[0] = meta[mrow + st_t];
                wv[1] = meta[mrow + st_t + 32];
                wv[2] = meta[mrow + st_t + 64];
                wv[3] = meta[mrow + st_t + 96];
                wv[4] = 0;
                if (st_t + 128 < 144) {
                    wv[4] = meta[mrow + st_t + 128];
                }
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(Vs + (h_kv_s * 128 + st_t * 4) + 0);
                    vsv[0 + 0] = _v4.x;
                    vsv[0 + 1] = _v4.y;
                    vsv[0 + 2] = _v4.z;
                    vsv[0 + 3] = _v4.w;
                }
                int head = __shfl_sync(0xFFFFFFFF, wv[0], 0);
                int qb0 = __shfl_sync(0xFFFFFFFF, wv[0], 1);
                int qb1 = __shfl_sync(0xFFFFFFFF, wv[0], 2);
                int mode_w = __shfl_sync(0xFFFFFFFF, wv[0], 3);
                int valid_s = __shfl_sync(0xFFFFFFFF, wv[0], 7);
                mbarrier_wait(meta_empty_addr, 1);
                if (elect_sync()) {
                }
                meta_smem[slot * 272 + st_t] = wv[0];
                meta_smem[slot * 272 + st_t + 32] = wv[1];
                meta_smem[slot * 272 + st_t + 64] = wv[2];
                meta_smem[slot * 272 + st_t + 96] = wv[3];
                if (st_t + 128 < 144) {
                    meta_smem[slot * 272 + st_t + 128] = wv[4];
                }
                vs_smem[slot * 272 + 144 + st_t * 4] = vsv[0];
                vs_smem[slot * 272 + 144 + st_t * 4 + 1] = vsv[1];
                vs_smem[slot * 272 + 144 + st_t * 4 + 2] = vsv[2];
                vs_smem[slot * 272 + 144 + st_t * 4 + 3] = vsv[3];
                mbarrier_arrive(meta_full_addr);
                int nt_st = __shfl_sync(0xFFFFFFFF, nt_st_ld, 0);
                int pool_base = g_ctas * tile_stride;
                #pragma unroll 1
                for (int ti = 1; ti < q_bound; ti++) {
                    int trow = row0 + ti;
                    if (nt_st <= ti) {
                        if (elect_sync()) {
                            int tq_fetch = atomicAdd(&dbg[62], 1);
                            tq_smem[ti & 1] = tq_fetch + pool_base;
                        }
                        __syncwarp();
                        trow = tq_smem[ti & 1];
                    }
                    int slot_0 = ti & 1;
                    int mrow_1 = trow * 144;
                    wv[0] = meta[mrow_1 + st_t];
                    wv[1] = meta[mrow_1 + st_t + 32];
                    wv[2] = meta[mrow_1 + st_t + 64];
                    wv[3] = meta[mrow_1 + st_t + 96];
                    wv[4] = 0;
                    if (st_t + 128 < 144) {
                        wv[4] = meta[mrow_1 + st_t + 128];
                    }
                    int head_2 = __shfl_sync(0xFFFFFFFF, wv[0], 0);
                    int qb0_3 = __shfl_sync(0xFFFFFFFF, wv[0], 1);
                    int qb1_4 = __shfl_sync(0xFFFFFFFF, wv[0], 2);
                    int mode_w_5 = __shfl_sync(0xFFFFFFFF, wv[0], 3);
                    int valid_s_6 = __shfl_sync(0xFFFFFFFF, wv[0], 7);
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(Vs + (qb1_4 * 128 + st_t * 4) + 0);
                        vsv[0 + 0] = _v4.x;
                        vsv[0 + 1] = _v4.y;
                        vsv[0 + 2] = _v4.z;
                        vsv[0 + 3] = _v4.w;
                    }
                    int _stage_meta_empty = (int)((unsigned int)ti % 2);
                    int _parity_meta_empty = (int)((unsigned int)ti / 2 % 2) ^ 1;
                    mbarrier_wait(meta_empty_addr + (_stage_meta_empty) * 8, _parity_meta_empty);
                    if (elect_sync()) {
                        if (ti > 0) {
                            if (valid_s_6 != 0) {
                                int _parity_q_empty = (int)((unsigned int)ti % 2) ^ 1;
                                mbarrier_wait(q_empty_addr, _parity_q_empty);
                                mbarrier_arrive_expect_tx(q_ready_addr, 8448);
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(q_smem_addr), "l"((&Q)), "r"(0), "r"(head_2 * seqlen_q + qb0_3 * 64),
                                       "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(qs_smem_addr), "l"((&Qs)), "r"(qb0_3 * 64), "r"(head_2),
                                       "r"(q_ready_addr), "l"(0x12F0000000000000ULL) : "memory");
                                if (mode_w_5 != 1) {
                                    mbarrier_arrive_expect_tx(q_ready_addr + 8, 8448);
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5;"
                                        :: "r"(q_smem_addr + 8192), "l"((&Q)), "r"(0), "r"((head_2 + 1) * seqlen_q + qb0_3 * 64),
                                           "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3}], [%4], %5;"
                                        :: "r"(qs_smem_addr + 256), "l"((&Qs)), "r"(qb0_3 * 64), "r"(head_2 + 1),
                                           "r"(q_ready_addr + 8), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                    meta_smem[slot_0 * 272 + st_t] = wv[0];
                    meta_smem[slot_0 * 272 + st_t + 32] = wv[1];
                    meta_smem[slot_0 * 272 + st_t + 64] = wv[2];
                    meta_smem[slot_0 * 272 + st_t + 96] = wv[3];
                    if (st_t + 128 < 144) {
                        meta_smem[slot_0 * 272 + st_t + 128] = wv[4];
                    }
                    vs_smem[slot_0 * 272 + 144 + st_t * 4] = vsv[0];
                    vs_smem[slot_0 * 272 + 144 + st_t * 4 + 1] = vsv[1];
                    vs_smem[slot_0 * 272 + 144 + st_t * 4 + 2] = vsv[2];
                    vs_smem[slot_0 * 272 + 144 + st_t * 4 + 3] = vsv[3];
                    mbarrier_arrive(meta_full_addr + ((int)((unsigned int)ti % 2)) * 8);
                    int valid_st = meta[trow * 144 + 7];
                    if (valid_st == 0) {
                        if (elect_sync()) {
                            __threadfence();
                            int tq_done = atomicAdd(&dbg[63], 1);
                            if (tq_done == g_ctas - 1) {
                                __threadfence();
                                dbg[62] = 0;
                                dbg[63] = 0;
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
                    int kvh1_k = kvh_k & 1;
                    unsigned long long pol_k = ((kvh1_k != 0) ? 1364590687093260288 : 1508705875169116160);
                    int h_head_k = hdr[hb + 2];
                    int n_hdr_k = hdr[hb + 4];
                    int kv_base_h = h_head_k * seqlen_k;
                    #pragma unroll
                    for (int hp = 0; hp < 3; hp++) {
                        if (n_hdr_k > hp) {
                            int hlap = 0;
                            hlap += hp / 3;
                            int _parity_k_empty = (int)((unsigned int)hlap % 2) ^ 1;
                            mbarrier_wait(k_empty_addr + (hp % 3) * 8, _parity_k_empty);
                            int hblk_a = hdr[hb + 5 + 2 * hp];
                            int hblk_b = hdr[hb + 6 + 2 * hp];
                            int has_b = 1;
                            if (hblk_b < 0) {
                                has_b = 0;
                            }
                            int ua = hblk_a;
                            int ub = hblk_b;
                            unsigned long long pol_a = pol_k;
                            unsigned long long pol_b = pol_k;
                            int k_dst = k_smem_addr + (unsigned int)(hp % 3 * 16384);
                            int ks_dst = ks_smem_addr + (unsigned int)(hp % 3 * 256);
                            mbarrier_arrive_expect_tx(k_full_addr + (hp % 3) * 8, 8208 * (1 + has_b));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(k_dst), "l"((&K)), "r"(0), "r"(kv_base_h + ua * 64),
                                   "r"(k_full_addr + (hp % 3) * 8), "l"(pol_a) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(ks_dst), "l"((&Ks)), "r"(ua * 4), "r"(h_head_k),
                                   "r"(k_full_addr + (hp % 3) * 8), "l"(pol_a) : "memory");
                            if (has_b != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(k_dst + 8192), "l"((&K)), "r"(0), "r"(kv_base_h + ub * 64),
                                       "r"(k_full_addr + (hp % 3) * 8), "l"(pol_b) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(ks_dst + 128), "l"((&Ks)), "r"(ub * 4), "r"(h_head_k),
                                       "r"(k_full_addr + (hp % 3) * 8), "l"(pol_b) : "memory");
                            }
                        }
                    }
                    int nt_k = q_bound;
                    #pragma unroll 1
                    for (int ti_1 = 0; ti_1 < nt_k; ti_1++) {
                        int slot_k = ti_1 & 1;
                        int mb_k = slot_k * 272;
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
                            int _parity_k_empty_1 = (int)((unsigned int)plap % 2) ^ 1;
                            mbarrier_wait(k_empty_addr + (stage) * 8, _parity_k_empty_1);
                            int sw = meta_smem[mb_k + 8 + i];
                            int blk_a = sw << 16 >> 16;
                            int blk_b = sw >> 16;
                            int has_b_1 = 1;
                            if (blk_b < 0) {
                                has_b_1 = 0;
                            }
                            int ua_1 = blk_a;
                            int ub_1 = blk_b;
                            unsigned long long pol_a_1 = pol_k;
                            unsigned long long pol_b_1 = pol_k;
                            int k_dst_1 = k_smem_addr + (unsigned int)(stage * 16384);
                            int ks_dst_1 = ks_smem_addr + (unsigned int)(stage * 256);
                            mbarrier_arrive_expect_tx(k_full_addr + (stage) * 8, 8208 * (1 + has_b_1));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(k_dst_1), "l"((&K)), "r"(0), "r"(kv_base + ua_1 * 64),
                                   "r"(k_full_addr + (stage) * 8), "l"(pol_a_1) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(ks_dst_1), "l"((&Ks)), "r"(ua_1 * 4), "r"(head_k),
                                   "r"(k_full_addr + (stage) * 8), "l"(pol_a_1) : "memory");
                            if (has_b_1 != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(k_dst_1 + 8192), "l"((&K)), "r"(0), "r"(kv_base + ub_1 * 64),
                                       "r"(k_full_addr + (stage) * 8), "l"(pol_b_1) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(ks_dst_1 + 128), "l"((&Ks)), "r"(ub_1 * 4), "r"(head_k),
                                       "r"(k_full_addr + (stage) * 8), "l"(pol_b_1) : "memory");
                            }
                            int ipk = i + 1;
                            if (ipk < n_seq_k) {
                                int swp = meta_smem[mb_k + 8 + ipk];
                                int pblk_a = swp << 16 >> 16;
                                int pblk_b = swp >> 16;
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(kv_base + pblk_a * 64)) : "memory");
                                if (pblk_b >= 0) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(kv_base + pblk_b * 64)) : "memory");
                                }
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
                    int kvh1_v = kvh_v & 1;
                    unsigned long long pol_v = ((kvh1_v != 0) ? 1364590687093260288 : 1508705875169116160);
                    int h_head_v = hdr[hb + 2];
                    int n_hdr_v = hdr[hb + 4];
                    int kv_base_hv = h_head_v * seqlen_k;
                    #pragma unroll
                    for (int hp_1 = 0; hp_1 < 3; hp_1++) {
                        if (n_hdr_v > hp_1) {
                            int hlap_1 = 0;
                            hlap_1 += hp_1 / 3;
                            int _parity_v_land_empty = (int)((unsigned int)hlap_1 % 2) ^ 1;
                            mbarrier_wait(v_land_empty_addr + (hp_1 % 3) * 8, _parity_v_land_empty);
                            int hvblk_a = hdr[hb + 5 + 2 * hp_1];
                            int hvblk_b = hdr[hb + 6 + 2 * hp_1];
                            int v_has_b = 1;
                            if (hvblk_b < 0) {
                                v_has_b = 0;
                            }
                            int vua = hvblk_a;
                            int vub = hvblk_b;
                            unsigned long long vpol_a = pol_v;
                            unsigned long long vpol_b = pol_v;
                            int vl_dst = vl_smem_addr + (unsigned int)(hp_1 % 3 * 16384);
                            mbarrier_arrive_expect_tx(v_land_full_addr + (hp_1 % 3) * 8, 8192 * (1 + v_has_b));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(vl_dst), "l"((&V)), "r"(0), "r"(kv_base_hv + vua * 64),
                                   "r"(v_land_full_addr + (hp_1 % 3) * 8), "l"(vpol_a) : "memory");
                            if (v_has_b != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(vl_dst + 8192), "l"((&V)), "r"(0), "r"(kv_base_hv + vub * 64),
                                       "r"(v_land_full_addr + (hp_1 % 3) * 8), "l"(vpol_b) : "memory");
                            }
                        }
                    }
                    int nt_v = q_bound;
                    #pragma unroll 1
                    for (int ti_2 = 0; ti_2 < nt_v; ti_2++) {
                        int slot_v = ti_2 & 1;
                        int mb_v = slot_v * 272;
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
                            int ls_v = pv % 3;
                            int pvlap = 0;
                            pvlap += pv / 3;
                            int _parity_v_land_empty_1 = (int)((unsigned int)pvlap % 2) ^ 1;
                            mbarrier_wait(v_land_empty_addr + (ls_v) * 8, _parity_v_land_empty_1);
                            int svw = meta_smem[mb_v + 8 + i_1];
                            int vblk_a = svw << 16 >> 16;
                            int vblk_b = svw >> 16;
                            int v_has_b_1 = 1;
                            if (vblk_b < 0) {
                                v_has_b_1 = 0;
                            }
                            int vua_1 = vblk_a;
                            int vub_1 = vblk_b;
                            unsigned long long vpol_a_1 = pol_v;
                            unsigned long long vpol_b_1 = pol_v;
                            int vl_dst_1 = vl_smem_addr + (unsigned int)(ls_v * 16384);
                            mbarrier_arrive_expect_tx(v_land_full_addr + (ls_v) * 8, 8192 * (1 + v_has_b_1));
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(vl_dst_1), "l"((&V)), "r"(0), "r"(kv_base_v + vua_1 * 64),
                                   "r"(v_land_full_addr + (ls_v) * 8), "l"(vpol_a_1) : "memory");
                            if (v_has_b_1 != 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3}], [%4], %5;"
                                    :: "r"(vl_dst_1 + 8192), "l"((&V)), "r"(0), "r"(kv_base_v + vub_1 * 64),
                                       "r"(v_land_full_addr + (ls_v) * 8), "l"(vpol_b_1) : "memory");
                            }
                            int ipv = i_1 + 1;
                            if (ipv < n_seq_v) {
                                int svp = meta_smem[mb_v + 8 + ipv];
                                int pvblk_a = svp << 16 >> 16;
                                int pvblk_b = svp >> 16;
                                asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(kv_base_v + pvblk_a * 64)) : "memory");
                                if (pvblk_b >= 0) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(kv_base_v + pvblk_b * 64)) : "memory");
                                }
                            }
                        }
                        gv += n_seq_v;
                        mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_2 % 2)) * 8);
                    }
                }
            }
            if (warp == 3) {
                asm volatile("barrier.sync 1, 384;" ::: "memory");
                int t_lane = lane;
                int t_mi = t_lane >> 3;
                int t_rr = t_lane & 7;
                int t_kl = t_mi * 8 + t_rr;
                int ld_row = t_kl * 128;
                int t_par = t_rr >> 2 ^ t_mi & 1;
                unsigned int sel_a = (unsigned int)(((t_lane < 16) ? 25632 : 30001));
                unsigned int sel_b = (unsigned int)(((t_lane < 16) ? 30001 : 25632));
                int t_dl = t_rr * 2 + t_par;
                int st_row = t_dl * 128;
                int t_mih = t_mi >> 1;
                int t_dl7 = t_dl & 7;
                unsigned int tw[16];
                unsigned int tpw[4];
                int gt = 0;
                int n_hdr_t = hdr[hb + 4];
                #pragma unroll
                for (int hp_2 = 0; hp_2 < 3; hp_2++) {
                    if (n_hdr_t > hp_2) {
                        int hblk_bt = hdr[hb + 6 + 2 * hp_2];
                        int hhas_bt = 1;
                        if (hblk_bt < 0) {
                            hhas_bt = 0;
                        }
                        int hlap_2 = 0;
                        hlap_2 += hp_2 / 3;
                        int _parity_v_land_full = (int)((unsigned int)hlap_2 % 2);
                        mbarrier_wait(v_land_full_addr + (hp_2 % 3) * 8, _parity_v_land_full);
                        int _parity_v_empty = (int)((unsigned int)hlap_2 % 2) ^ 1;
                        mbarrier_wait(v_empty_addr + (hp_2 % 3) * 8, _parity_v_empty);
                        int land_off = hp_2 % 3 * 16384;
                        int vt_off = hp_2 % 3 * 16384;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 0) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 1) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 2) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 3) << 4)))
                            : "memory");
                        uint32_t _prmt_2;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_2) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_2;
                        uint32_t _prmt_3;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_3) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_3;
                        uint32_t _prmt_4;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_4) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_4;
                        uint32_t _prmt_5;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_5) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_5;
                        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_7;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_7) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_7;
                        uint32_t _prmt_8;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_8) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_8;
                        uint32_t _prmt_9;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_9) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_9;
                        uint32_t _prmt_10;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_10) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_10;
                        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 2048 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_12;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_12) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_12;
                        uint32_t _prmt_13;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_13) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_13;
                        uint32_t _prmt_14;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_14) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_14;
                        uint32_t _prmt_15;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_15) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_15;
                        uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 4096 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_17;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_17) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_17;
                        uint32_t _prmt_18;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_18) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_18;
                        uint32_t _prmt_19;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_19) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_19;
                        uint32_t _prmt_20;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_20) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_20;
                        uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 6144 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 4) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 5) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 6) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + ld_row + ((t_rr ^ 7) << 4)))
                            : "memory");
                        uint32_t _prmt_22;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_22) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_22;
                        uint32_t _prmt_23;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_23) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_23;
                        uint32_t _prmt_24;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_24) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_24;
                        uint32_t _prmt_25;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_25) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_25;
                        uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 8192 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_27;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_27) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_27;
                        uint32_t _prmt_28;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_28) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_28;
                        uint32_t _prmt_29;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_29) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_29;
                        uint32_t _prmt_30;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_30) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_30;
                        uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 10240 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_32;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_32) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_32;
                        uint32_t _prmt_33;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_33) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_33;
                        uint32_t _prmt_34;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_34) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_34;
                        uint32_t _prmt_35;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_35) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_35;
                        uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 12288 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_37;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_37) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_37;
                        uint32_t _prmt_38;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_38) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_38;
                        uint32_t _prmt_39;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_39) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_39;
                        uint32_t _prmt_40;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_40) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_40;
                        uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 14336 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 0) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 1) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 2) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 3) << 4)))
                            : "memory");
                        uint32_t _prmt_42;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_42) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_42;
                        uint32_t _prmt_43;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_43) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_43;
                        uint32_t _prmt_44;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_44) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_44;
                        uint32_t _prmt_45;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_45) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_45;
                        uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_47;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_47) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_47;
                        uint32_t _prmt_48;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_48) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_48;
                        uint32_t _prmt_49;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_49) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_49;
                        uint32_t _prmt_50;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_50) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_50;
                        uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 2048 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_52;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_52) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_52;
                        uint32_t _prmt_53;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_53) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_53;
                        uint32_t _prmt_54;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_54) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_54;
                        uint32_t _prmt_55;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_55) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_55;
                        uint32_t _stmatrix_addr_56 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 4096 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_56), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_57;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_57) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_57;
                        uint32_t _prmt_58;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_58) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_58;
                        uint32_t _prmt_59;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_59) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_59;
                        uint32_t _prmt_60;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_60) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_60;
                        uint32_t _stmatrix_addr_61 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 6144 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_61), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 4) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 5) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 6) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off + 4096 + ld_row + ((t_rr ^ 7) << 4)))
                            : "memory");
                        uint32_t _prmt_62;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_62) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_62;
                        uint32_t _prmt_63;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_63) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_63;
                        uint32_t _prmt_64;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_64) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_64;
                        uint32_t _prmt_65;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_65) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_65;
                        uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 8192 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_67;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_67) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_67;
                        uint32_t _prmt_68;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_68) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_68;
                        uint32_t _prmt_69;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_69) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_69;
                        uint32_t _prmt_70;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_70) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_70;
                        uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 10240 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_72;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_72) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_72;
                        uint32_t _prmt_73;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_73) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_73;
                        uint32_t _prmt_74;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_74) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_74;
                        uint32_t _prmt_75;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_75) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_75;
                        uint32_t _stmatrix_addr_76 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 12288 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_77;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_77) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_77;
                        uint32_t _prmt_78;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_78) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_78;
                        uint32_t _prmt_79;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_79) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_79;
                        uint32_t _prmt_80;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_80) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_80;
                        uint32_t _stmatrix_addr_81 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 14336 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_81), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        if (hhas_bt != 0) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 0) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 1) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 2) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 3) << 4)))
                                : "memory");
                            uint32_t _prmt_82;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_82) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_82;
                            uint32_t _prmt_83;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_83) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_83;
                            uint32_t _prmt_84;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_84) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_84;
                            uint32_t _prmt_85;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_85) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_85;
                            uint32_t _stmatrix_addr_86 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_86), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_87;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_87) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_87;
                            uint32_t _prmt_88;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_88) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_88;
                            uint32_t _prmt_89;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_89) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_89;
                            uint32_t _prmt_90;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_90) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_90;
                            uint32_t _stmatrix_addr_91 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 2048 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_91), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_92;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_92) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_92;
                            uint32_t _prmt_93;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_93) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_93;
                            uint32_t _prmt_94;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_94) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_94;
                            uint32_t _prmt_95;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_95) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_95;
                            uint32_t _stmatrix_addr_96 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 4096 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_96), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_97;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_97) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_97;
                            uint32_t _prmt_98;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_98) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_98;
                            uint32_t _prmt_99;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_99) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_99;
                            uint32_t _prmt_100;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_100) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_100;
                            uint32_t _stmatrix_addr_101 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 6144 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_101), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 4) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 5) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 6) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + ld_row + ((t_rr ^ 7) << 4)))
                                : "memory");
                            uint32_t _prmt_102;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_102) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_102;
                            uint32_t _prmt_103;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_103) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_103;
                            uint32_t _prmt_104;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_104) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_104;
                            uint32_t _prmt_105;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_105) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_105;
                            uint32_t _stmatrix_addr_106 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 8192 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_106), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_107;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_107) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_107;
                            uint32_t _prmt_108;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_108) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_108;
                            uint32_t _prmt_109;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_109) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_109;
                            uint32_t _prmt_110;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_110) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_110;
                            uint32_t _stmatrix_addr_111 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 10240 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_111), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_112;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_112) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_112;
                            uint32_t _prmt_113;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_113) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_113;
                            uint32_t _prmt_114;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_114) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_114;
                            uint32_t _prmt_115;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_115) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_115;
                            uint32_t _stmatrix_addr_116 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 12288 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_116), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_117;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_117) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_117;
                            uint32_t _prmt_118;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_118) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_118;
                            uint32_t _prmt_119;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_119) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_119;
                            uint32_t _prmt_120;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_120) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_120;
                            uint32_t _stmatrix_addr_121 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 14336 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_121), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 0) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 1) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 2) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 3) << 4)))
                                : "memory");
                            uint32_t _prmt_122;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_122) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_122;
                            uint32_t _prmt_123;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_123) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_123;
                            uint32_t _prmt_124;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_124) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_124;
                            uint32_t _prmt_125;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_125) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_125;
                            uint32_t _stmatrix_addr_126 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_126), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_127;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_127) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_127;
                            uint32_t _prmt_128;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_128) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_128;
                            uint32_t _prmt_129;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_129) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_129;
                            uint32_t _prmt_130;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_130) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_130;
                            uint32_t _stmatrix_addr_131 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 2048 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_131), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_132;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_132) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_132;
                            uint32_t _prmt_133;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_133) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_133;
                            uint32_t _prmt_134;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_134) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_134;
                            uint32_t _prmt_135;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_135) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_135;
                            uint32_t _stmatrix_addr_136 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 4096 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_136), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_137;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_137) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_137;
                            uint32_t _prmt_138;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_138) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_138;
                            uint32_t _prmt_139;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_139) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_139;
                            uint32_t _prmt_140;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_140) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_140;
                            uint32_t _stmatrix_addr_141 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 6144 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_141), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 4) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 5) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 6) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off + 8192 + 4096 + ld_row + ((t_rr ^ 7) << 4)))
                                : "memory");
                            uint32_t _prmt_142;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_142) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_142;
                            uint32_t _prmt_143;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_143) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_143;
                            uint32_t _prmt_144;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_144) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_144;
                            uint32_t _prmt_145;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_145) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_145;
                            uint32_t _stmatrix_addr_146 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 8192 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_146), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_147;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_147) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_147;
                            uint32_t _prmt_148;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_148) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_148;
                            uint32_t _prmt_149;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_149) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_149;
                            uint32_t _prmt_150;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_150) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_150;
                            uint32_t _stmatrix_addr_151 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 10240 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_151), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_152;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_152) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_152;
                            uint32_t _prmt_153;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_153) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_153;
                            uint32_t _prmt_154;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_154) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_154;
                            uint32_t _prmt_155;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_155) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_155;
                            uint32_t _stmatrix_addr_156 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 12288 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_156), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_157;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_157) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_157;
                            uint32_t _prmt_158;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_158) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_158;
                            uint32_t _prmt_159;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_159) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_159;
                            uint32_t _prmt_160;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_160) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_160;
                            uint32_t _stmatrix_addr_161 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off + 14336 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_161), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(v_full_addr + (hp_2 % 3) * 8);
                        mbarrier_arrive(v_land_empty_addr + (hp_2 % 3) * 8);
                    }
                }
                int nt_t = q_bound;
                #pragma unroll 1
                for (int ti_3 = 0; ti_3 < nt_t; ti_3++) {
                    int slot_t = ti_3 & 1;
                    int mb_t = slot_t * 272;
                    int _stage_meta_full_2 = (int)((unsigned int)ti_3 % 2);
                    int _parity_meta_full_2 = (int)((unsigned int)ti_3 / 2 % 2);
                    mbarrier_wait(meta_full_addr + (_stage_meta_full_2) * 8, _parity_meta_full_2);
                    int valid_t = __shfl_sync(0xFFFFFFFF, meta_smem[mb_t + 7], 0);
                    if (valid_t == 0) {
                        break;
                    }
                    int n_seq_t = __shfl_sync(0xFFFFFFFF, meta_smem[mb_t + 4], 0);
                    int i0_t = 0;
                    if (ti_3 == 0) {
                        i0_t = n_hdr_t;
                    }
                    #pragma unroll 1
                    for (int i_2 = i0_t; i_2 < n_seq_t; i_2++) {
                        int pt = gt + i_2;
                        int swt = __shfl_sync(0xFFFFFFFF, meta_smem[mb_t + 8 + i_2], 0);
                        int blk_bt = swt >> 16;
                        int has_bt = 1;
                        if (blk_bt < 0) {
                            has_bt = 0;
                        }
                        int tlap = 0;
                        tlap += pt / 3;
                        int _parity_v_land_full_1 = (int)((unsigned int)tlap % 2);
                        mbarrier_wait(v_land_full_addr + (pt % 3) * 8, _parity_v_land_full_1);
                        int _parity_v_empty_1 = (int)((unsigned int)tlap % 2) ^ 1;
                        mbarrier_wait(v_empty_addr + (pt % 3) * 8, _parity_v_empty_1);
                        int land_off_1 = pt % 3 * 16384;
                        int vt_off_1 = pt % 3 * 16384;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 0) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 1) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 2) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 3) << 4)))
                            : "memory");
                        uint32_t _prmt_162;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_162) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_162;
                        uint32_t _prmt_163;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_163) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_163;
                        uint32_t _prmt_164;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_164) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_164;
                        uint32_t _prmt_165;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_165) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_165;
                        uint32_t _stmatrix_addr_166 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_166), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_167;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_167) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_167;
                        uint32_t _prmt_168;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_168) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_168;
                        uint32_t _prmt_169;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_169) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_169;
                        uint32_t _prmt_170;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_170) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_170;
                        uint32_t _stmatrix_addr_171 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 2048 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_171), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_172;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_172) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_172;
                        uint32_t _prmt_173;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_173) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_173;
                        uint32_t _prmt_174;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_174) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_174;
                        uint32_t _prmt_175;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_175) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_175;
                        uint32_t _stmatrix_addr_176 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 4096 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_176), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_177;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_177) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_177;
                        uint32_t _prmt_178;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_178) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_178;
                        uint32_t _prmt_179;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_179) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_179;
                        uint32_t _prmt_180;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_180) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_180;
                        uint32_t _stmatrix_addr_181 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 6144 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_181), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 4) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 5) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 6) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + ld_row + ((t_rr ^ 7) << 4)))
                            : "memory");
                        uint32_t _prmt_182;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_182) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_182;
                        uint32_t _prmt_183;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_183) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_183;
                        uint32_t _prmt_184;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_184) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_184;
                        uint32_t _prmt_185;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_185) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_185;
                        uint32_t _stmatrix_addr_186 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 8192 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_186), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_187;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_187) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_187;
                        uint32_t _prmt_188;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_188) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_188;
                        uint32_t _prmt_189;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_189) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_189;
                        uint32_t _prmt_190;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_190) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_190;
                        uint32_t _stmatrix_addr_191 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 10240 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_191), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_192;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_192) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_192;
                        uint32_t _prmt_193;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_193) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_193;
                        uint32_t _prmt_194;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_194) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_194;
                        uint32_t _prmt_195;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_195) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_195;
                        uint32_t _stmatrix_addr_196 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 12288 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_196), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_197;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_197) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_197;
                        uint32_t _prmt_198;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_198) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_198;
                        uint32_t _prmt_199;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_199) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_199;
                        uint32_t _prmt_200;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_200) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_200;
                        uint32_t _stmatrix_addr_201 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 14336 + st_row + ((t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_201), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 0) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 1) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 2) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 3) << 4)))
                            : "memory");
                        uint32_t _prmt_202;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_202) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_202;
                        uint32_t _prmt_203;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_203) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_203;
                        uint32_t _prmt_204;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_204) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_204;
                        uint32_t _prmt_205;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_205) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_205;
                        uint32_t _stmatrix_addr_206 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_206), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_207;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_207) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_207;
                        uint32_t _prmt_208;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_208) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_208;
                        uint32_t _prmt_209;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_209) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_209;
                        uint32_t _prmt_210;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_210) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_210;
                        uint32_t _stmatrix_addr_211 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 2048 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_211), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_212;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_212) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_212;
                        uint32_t _prmt_213;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_213) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_213;
                        uint32_t _prmt_214;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_214) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_214;
                        uint32_t _prmt_215;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_215) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_215;
                        uint32_t _stmatrix_addr_216 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 4096 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_216), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_217;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_217) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_217;
                        uint32_t _prmt_218;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_218) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_218;
                        uint32_t _prmt_219;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_219) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_219;
                        uint32_t _prmt_220;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_220) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_220;
                        uint32_t _stmatrix_addr_221 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 6144 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_221), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 4) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 5) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 6) << 4)))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                            : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 4096 + ld_row + ((t_rr ^ 7) << 4)))
                            : "memory");
                        uint32_t _prmt_222;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_222) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                        tpw[0] = _prmt_222;
                        uint32_t _prmt_223;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_223) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                        tpw[1] = _prmt_223;
                        uint32_t _prmt_224;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_224) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                        tpw[2] = _prmt_224;
                        uint32_t _prmt_225;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_225) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                        tpw[3] = _prmt_225;
                        uint32_t _stmatrix_addr_226 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 8192 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_226), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_227;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_227) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                        tpw[0] = _prmt_227;
                        uint32_t _prmt_228;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_228) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                        tpw[1] = _prmt_228;
                        uint32_t _prmt_229;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_229) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                        tpw[2] = _prmt_229;
                        uint32_t _prmt_230;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_230) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                        tpw[3] = _prmt_230;
                        uint32_t _stmatrix_addr_231 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 10240 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_231), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_232;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_232) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                        tpw[0] = _prmt_232;
                        uint32_t _prmt_233;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_233) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                        tpw[1] = _prmt_233;
                        uint32_t _prmt_234;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_234) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                        tpw[2] = _prmt_234;
                        uint32_t _prmt_235;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_235) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                        tpw[3] = _prmt_235;
                        uint32_t _stmatrix_addr_236 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 12288 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_236), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        uint32_t _prmt_237;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_237) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                        tpw[0] = _prmt_237;
                        uint32_t _prmt_238;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_238) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                        tpw[1] = _prmt_238;
                        uint32_t _prmt_239;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_239) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                        tpw[2] = _prmt_239;
                        uint32_t _prmt_240;
                        asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_240) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                        tpw[3] = _prmt_240;
                        uint32_t _stmatrix_addr_241 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 14336 + st_row + ((2 + t_mih ^ t_dl7) << 4)));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_241), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                            : "memory");
                        if (has_bt != 0) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 0) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 1) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 2) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 3) << 4)))
                                : "memory");
                            uint32_t _prmt_242;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_242) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_242;
                            uint32_t _prmt_243;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_243) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_243;
                            uint32_t _prmt_244;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_244) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_244;
                            uint32_t _prmt_245;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_245) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_245;
                            uint32_t _stmatrix_addr_246 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_246), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_247;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_247) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_247;
                            uint32_t _prmt_248;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_248) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_248;
                            uint32_t _prmt_249;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_249) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_249;
                            uint32_t _prmt_250;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_250) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_250;
                            uint32_t _stmatrix_addr_251 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 2048 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_251), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_252;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_252) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_252;
                            uint32_t _prmt_253;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_253) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_253;
                            uint32_t _prmt_254;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_254) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_254;
                            uint32_t _prmt_255;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_255) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_255;
                            uint32_t _stmatrix_addr_256 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 4096 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_256), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_257;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_257) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_257;
                            uint32_t _prmt_258;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_258) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_258;
                            uint32_t _prmt_259;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_259) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_259;
                            uint32_t _prmt_260;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_260) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_260;
                            uint32_t _stmatrix_addr_261 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 6144 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_261), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 4) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 5) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 6) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + ld_row + ((t_rr ^ 7) << 4)))
                                : "memory");
                            uint32_t _prmt_262;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_262) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_262;
                            uint32_t _prmt_263;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_263) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_263;
                            uint32_t _prmt_264;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_264) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_264;
                            uint32_t _prmt_265;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_265) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_265;
                            uint32_t _stmatrix_addr_266 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 8192 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_266), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_267;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_267) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_267;
                            uint32_t _prmt_268;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_268) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_268;
                            uint32_t _prmt_269;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_269) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_269;
                            uint32_t _prmt_270;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_270) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_270;
                            uint32_t _stmatrix_addr_271 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 10240 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_271), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_272;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_272) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_272;
                            uint32_t _prmt_273;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_273) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_273;
                            uint32_t _prmt_274;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_274) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_274;
                            uint32_t _prmt_275;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_275) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_275;
                            uint32_t _stmatrix_addr_276 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 12288 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_276), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_277;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_277) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_277;
                            uint32_t _prmt_278;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_278) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_278;
                            uint32_t _prmt_279;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_279) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_279;
                            uint32_t _prmt_280;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_280) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_280;
                            uint32_t _stmatrix_addr_281 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 14336 + st_row + ((4 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_281), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 0) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 1) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 2) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 3) << 4)))
                                : "memory");
                            uint32_t _prmt_282;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_282) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_282;
                            uint32_t _prmt_283;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_283) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_283;
                            uint32_t _prmt_284;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_284) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_284;
                            uint32_t _prmt_285;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_285) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_285;
                            uint32_t _stmatrix_addr_286 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_286), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_287;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_287) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_287;
                            uint32_t _prmt_288;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_288) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_288;
                            uint32_t _prmt_289;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_289) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_289;
                            uint32_t _prmt_290;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_290) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_290;
                            uint32_t _stmatrix_addr_291 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 2048 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_291), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_292;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_292) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_292;
                            uint32_t _prmt_293;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_293) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_293;
                            uint32_t _prmt_294;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_294) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_294;
                            uint32_t _prmt_295;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_295) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_295;
                            uint32_t _stmatrix_addr_296 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 4096 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_296), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_297;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_297) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_297;
                            uint32_t _prmt_298;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_298) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_298;
                            uint32_t _prmt_299;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_299) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_299;
                            uint32_t _prmt_300;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_300) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_300;
                            uint32_t _stmatrix_addr_301 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 6144 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_301), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[0]), "=r"(tw[1]), "=r"(tw[2]), "=r"(tw[3])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 4) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[4]), "=r"(tw[5]), "=r"(tw[6]), "=r"(tw[7])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 5) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[8]), "=r"(tw[9]), "=r"(tw[10]), "=r"(tw[11])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 6) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(tw[12]), "=r"(tw[13]), "=r"(tw[14]), "=r"(tw[15])
                                : "r"(vl_smem_addr + (unsigned int)(land_off_1 + 8192 + 4096 + ld_row + ((t_rr ^ 7) << 4)))
                                : "memory");
                            uint32_t _prmt_302;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_302) : "r"(tw[0]), "r"(tw[1]), "r"(sel_a));
                            tpw[0] = _prmt_302;
                            uint32_t _prmt_303;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_303) : "r"(tw[0]), "r"(tw[1]), "r"(sel_b));
                            tpw[1] = _prmt_303;
                            uint32_t _prmt_304;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_304) : "r"(tw[2]), "r"(tw[3]), "r"(sel_a));
                            tpw[2] = _prmt_304;
                            uint32_t _prmt_305;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_305) : "r"(tw[2]), "r"(tw[3]), "r"(sel_b));
                            tpw[3] = _prmt_305;
                            uint32_t _stmatrix_addr_306 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 8192 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_306), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_307;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_307) : "r"(tw[4]), "r"(tw[5]), "r"(sel_a));
                            tpw[0] = _prmt_307;
                            uint32_t _prmt_308;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_308) : "r"(tw[4]), "r"(tw[5]), "r"(sel_b));
                            tpw[1] = _prmt_308;
                            uint32_t _prmt_309;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_309) : "r"(tw[6]), "r"(tw[7]), "r"(sel_a));
                            tpw[2] = _prmt_309;
                            uint32_t _prmt_310;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_310) : "r"(tw[6]), "r"(tw[7]), "r"(sel_b));
                            tpw[3] = _prmt_310;
                            uint32_t _stmatrix_addr_311 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 10240 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_311), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_312;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_312) : "r"(tw[8]), "r"(tw[9]), "r"(sel_a));
                            tpw[0] = _prmt_312;
                            uint32_t _prmt_313;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_313) : "r"(tw[8]), "r"(tw[9]), "r"(sel_b));
                            tpw[1] = _prmt_313;
                            uint32_t _prmt_314;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_314) : "r"(tw[10]), "r"(tw[11]), "r"(sel_a));
                            tpw[2] = _prmt_314;
                            uint32_t _prmt_315;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_315) : "r"(tw[10]), "r"(tw[11]), "r"(sel_b));
                            tpw[3] = _prmt_315;
                            uint32_t _stmatrix_addr_316 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 12288 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_316), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                            uint32_t _prmt_317;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_317) : "r"(tw[12]), "r"(tw[13]), "r"(sel_a));
                            tpw[0] = _prmt_317;
                            uint32_t _prmt_318;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_318) : "r"(tw[12]), "r"(tw[13]), "r"(sel_b));
                            tpw[1] = _prmt_318;
                            uint32_t _prmt_319;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_319) : "r"(tw[14]), "r"(tw[15]), "r"(sel_a));
                            tpw[2] = _prmt_319;
                            uint32_t _prmt_320;
                            asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_320) : "r"(tw[14]), "r"(tw[15]), "r"(sel_b));
                            tpw[3] = _prmt_320;
                            uint32_t _stmatrix_addr_321 = static_cast<uint32_t>(vt_smem_addr + (unsigned int)(vt_off_1 + 14336 + st_row + ((6 + t_mih ^ t_dl7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_321), "r"(*reinterpret_cast<const uint32_t*>(&tpw[0])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[1])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[2])), "r"(*reinterpret_cast<const uint32_t*>(&tpw[3]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(v_full_addr + (pt % 3) * 8);
                        mbarrier_arrive(v_land_empty_addr + (pt % 3) * 8);
                    }
                    gt += n_seq_t;
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_3 % 2)) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: main ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
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
            smem_v11[tid_c] = 0.0f;
            smem_v11[256 + tid_c] = 0.0f;
            smem_v11[512 + tid_c] = 0.0f;
            smem_v11[768 + tid_c] = 0.0f;
            smem_v11[1024 + tid_c] = 0.0f;
            smem_v11[1280 + tid_c] = 0.0f;
            smem_v11[1536 + tid_c] = 0.0f;
            smem_v11[1792 + tid_c] = 0.0f;
            smem_v11[2048 + tid_c] = 0.0f;
            smem_v11[2304 + tid_c] = 0.0f;
            smem_v11[2560 + tid_c] = 0.0f;
            smem_v11[2816 + tid_c] = 0.0f;
            smem_v11[3072 + tid_c] = 0.0f;
            smem_v11[3328 + tid_c] = 0.0f;
            smem_v11[3584 + tid_c] = 0.0f;
            smem_v11[3840 + tid_c] = 0.0f;
            smem_v12[tid_c] = 0.0f;
            smem_v12[256 + tid_c] = 0.0f;
            smem_v12[512 + tid_c] = 0.0f;
            smem_v12[768 + tid_c] = 0.0f;
            smem_v12[1024 + tid_c] = 0.0f;
            smem_v12[1280 + tid_c] = 0.0f;
            smem_v12[1536 + tid_c] = 0.0f;
            smem_v12[1792 + tid_c] = 0.0f;
            smem_v12[2048 + tid_c] = 0.0f;
            smem_v12[2304 + tid_c] = 0.0f;
            smem_v12[2560 + tid_c] = 0.0f;
            smem_v12[2816 + tid_c] = 0.0f;
            smem_v12[3072 + tid_c] = 0.0f;
            smem_v12[3328 + tid_c] = 0.0f;
            smem_v12[3584 + tid_c] = 0.0f;
            smem_v12[3840 + tid_c] = 0.0f;
            smem_v13[tid_c] = 0.0f;
            smem_v13[256 + tid_c] = 0.0f;
            smem_v13[512 + tid_c] = 0.0f;
            smem_v13[768 + tid_c] = 0.0f;
            smem_v13[1024 + tid_c] = 0.0f;
            smem_v13[1280 + tid_c] = 0.0f;
            smem_v13[1536 + tid_c] = 0.0f;
            smem_v13[1792 + tid_c] = 0.0f;
            smem_v13[2048 + tid_c] = 0.0f;
            smem_v13[2304 + tid_c] = 0.0f;
            smem_v13[2560 + tid_c] = 0.0f;
            smem_v13[2816 + tid_c] = 0.0f;
            smem_v13[3072 + tid_c] = 0.0f;
            smem_v13[3328 + tid_c] = 0.0f;
            smem_v13[3584 + tid_c] = 0.0f;
            smem_v13[3840 + tid_c] = 0.0f;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.arrive 1, 384;" ::: "memory");
            float d_o[64];
            float d_qk[64];
            unsigned int p_e4m3[16];
            float poly_tf[2];
            int poly_ti[2];
            unsigned int q_frag[16];
            float q_log2[2];
            float qk_sc[16];
            float row_max0 = -CUDART_INF_F;
            float row_max1 = -CUDART_INF_F;
            float row_sum0 = 0.0f;
            float row_sum1 = 0.0f;
            int qt = lane & 3;
            int gbase = 0;
            int prev = -1;
            int g_ctas_m = gridDim.x;
            int nt_m = tile_stride + 256 * g_ctas_m + 1;
            #pragma unroll 1
            for (int ti_4 = 0; ti_4 < nt_m; ti_4++) {
                int slot_m = ti_4 & 1;
                int mbase = slot_m * 272;
                int _stage_meta_full_3 = (int)((unsigned int)ti_4 % 2);
                int _parity_meta_full_3 = (int)((unsigned int)ti_4 / 2 % 2);
                mbarrier_wait(meta_full_addr + (_stage_meta_full_3) * 8, _parity_meta_full_3);
                int valid_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 7], 0);
                if (valid_m == 0) {
                    break;
                }
                int head_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase], 0);
                int qb0_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 1], 0);
                int n_seq_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 4], 0);
                int n_own = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + cwg], 0);
                int n_other = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 5 + 1 - cwg], 0);
                int mode_m = __shfl_sync(0xFFFFFFFF, meta_smem[mbase + 3], 0);
                int is_split = mode_m == 1;
                int my_qb = qb0_m;
                int head_o = ((is_split != 0) ? head_m : head_m + cwg);
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
                int _parity_q_ready = (int)((unsigned int)ti_4 % 2);
                mbarrier_wait(q_ready_addr + (qs) * 8, _parity_q_ready);
                unsigned int q_base = q_smem_addr + (unsigned int)(qs * 8192);
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[0])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[1])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[2])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(16 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[3])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(16 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[4])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(32 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[5])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(32 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[6])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(48 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[7])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(48 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[8])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(64 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[9])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(64 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[10])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(80 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[11])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(80 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[12])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(96 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[13])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(96 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[14])) : "r"((q_base + (unsigned int)(m0_local * 128) + ((unsigned int)(112 + 4 * qt) ^ (q_base + (unsigned int)(m0_local * 128) >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&q_frag[15])) : "r"((q_base + (unsigned int)(m1_local * 128) + ((unsigned int)(112 + 4 * qt) ^ (q_base + (unsigned int)(m1_local * 128) >> 7 & 7) << 4))));
                q_log2[0] = qs_smem[qs * 64 + m0_local] * scale_log2;
                q_log2[1] = qs_smem[qs * 64 + m1_local] * scale_log2;
                if (elect_sync()) {
                    mbarrier_arrive(q_empty_addr);
                }
                int cur_pos = 0;
                int cur_has2 = 1;
                if (n_own > 0) {
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
                    int kpos0 = 0;
                    kpos0 += pos0;
                    int klap0 = 0;
                    klap0 += pos0 / 3;
                    cur_pos = pos0;
                    cur_has2 = has2_0;
                    int stage_0 = pos0 % 3;
                    int _parity_k_full_1 = (int)((unsigned int)klap0 % 2);
                    mbarrier_wait(k_full_addr + (kpos0 % 3) * 8, _parity_k_full_1);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_0 = (((uint64_t)(((k_smem_addr + (unsigned int)(stage_0 * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(wgmma_b)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(wgmma_b + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(wgmma_b + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                        : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(wgmma_b + 6)
                        : "memory");
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                    float ksg_7_0 = ks_smem[stage_0 * 64];
                    qk_sc[0] = q_log2[0] * ksg_7_0;
                    qk_sc[8] = q_log2[1] * ksg_7_0;
                    float ksg_7_1 = ks_smem[stage_0 * 64 + 1];
                    qk_sc[1] = q_log2[0] * ksg_7_1;
                    qk_sc[9] = q_log2[1] * ksg_7_1;
                    float ksg_7_2 = ks_smem[stage_0 * 64 + 2];
                    qk_sc[2] = q_log2[0] * ksg_7_2;
                    qk_sc[10] = q_log2[1] * ksg_7_2;
                    float ksg_7_3 = ks_smem[stage_0 * 64 + 3];
                    qk_sc[3] = q_log2[0] * ksg_7_3;
                    qk_sc[11] = q_log2[1] * ksg_7_3;
                    float ksg_7_4 = ks_smem[stage_0 * 64 + 32];
                    qk_sc[4] = q_log2[0] * ksg_7_4;
                    qk_sc[12] = q_log2[1] * ksg_7_4;
                    float ksg_7_5 = ks_smem[stage_0 * 64 + 32 + 1];
                    qk_sc[5] = q_log2[0] * ksg_7_5;
                    qk_sc[13] = q_log2[1] * ksg_7_5;
                    float ksg_7_6 = ks_smem[stage_0 * 64 + 32 + 2];
                    qk_sc[6] = q_log2[0] * ksg_7_6;
                    qk_sc[14] = q_log2[1] * ksg_7_6;
                    float ksg_7_7 = ks_smem[stage_0 * 64 + 32 + 3];
                    qk_sc[7] = q_log2[0] * ksg_7_7;
                    qk_sc[15] = q_log2[1] * ksg_7_7;
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (kpos0 % 3) * 8);
                    }
                    float f_max0 = -CUDART_INF_F;
                    float f_max1 = -CUDART_INF_F;
                    float m0 = -CUDART_INF_F;
                    float m1 = -CUDART_INF_F;
                    if (scale_log2 >= 0.0f) {
                        float max_1 = fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5]));
                        float max_1_0 = fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]));
                        float max_1_1 = fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21]));
                        float max_1_2 = fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]));
                        float max_1_3 = fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37]));
                        float max_1_4 = fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]));
                        float max_1_5 = fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53]));
                        float max_1_6 = fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]));
                        float max_1_7 = fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7]));
                        float max_1_8 = fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]));
                        float max_1_9 = fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23]));
                        float max_1_10 = fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]));
                        float max_1_11 = fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39]));
                        float max_1_12 = fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]));
                        float max_1_13 = fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55]));
                        float max_1_14 = fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]));
                        float max_1_15 = fmaxf(fmaxf(max_1 * qk_sc[0], max_1_0 * qk_sc[1]), fmaxf(max_1_1 * qk_sc[2], max_1_2 * qk_sc[3]));
                        float max_1_16 = fmaxf(fmaxf(max_1_3 * qk_sc[4], max_1_4 * qk_sc[5]), fmaxf(max_1_5 * qk_sc[6], max_1_6 * qk_sc[7]));
                        float max_1_17 = fmaxf(fmaxf(max_1_7 * qk_sc[8], max_1_8 * qk_sc[9]), fmaxf(max_1_9 * qk_sc[10], max_1_10 * qk_sc[11]));
                        float max_1_18 = fmaxf(fmaxf(max_1_11 * qk_sc[12], max_1_12 * qk_sc[13]), fmaxf(max_1_13 * qk_sc[14], max_1_14 * qk_sc[15]));
                        m0 = ((has2_0 != 0) ? fmaxf(max_1_15, max_1_16) : max_1_15);
                        m1 = ((has2_0 != 0) ? fmaxf(max_1_17, max_1_18) : max_1_17);
                    } else {
                        float min_1 = fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5]));
                        float min_1_0 = fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]));
                        float min_1_1 = fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21]));
                        float min_1_2 = fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]));
                        float min_1_3 = fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37]));
                        float min_1_4 = fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]));
                        float min_1_5 = fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53]));
                        float min_1_6 = fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]));
                        float min_1_7 = fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7]));
                        float min_1_8 = fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]));
                        float min_1_9 = fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23]));
                        float min_1_10 = fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]));
                        float min_1_11 = fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39]));
                        float min_1_12 = fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]));
                        float min_1_13 = fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55]));
                        float min_1_14 = fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]));
                        float max_1_19 = fmaxf(fmaxf(min_1 * qk_sc[0], min_1_0 * qk_sc[1]), fmaxf(min_1_1 * qk_sc[2], min_1_2 * qk_sc[3]));
                        float max_1_19_1 = fmaxf(fmaxf(min_1_3 * qk_sc[4], min_1_4 * qk_sc[5]), fmaxf(min_1_5 * qk_sc[6], min_1_6 * qk_sc[7]));
                        float max_1_20 = fmaxf(fmaxf(min_1_7 * qk_sc[8], min_1_8 * qk_sc[9]), fmaxf(min_1_9 * qk_sc[10], min_1_10 * qk_sc[11]));
                        float max_1_21 = fmaxf(fmaxf(min_1_11 * qk_sc[12], min_1_12 * qk_sc[13]), fmaxf(min_1_13 * qk_sc[14], min_1_14 * qk_sc[15]));
                        m0 = ((has2_0 != 0) ? fmaxf(max_1_19, max_1_19_1) : max_1_19);
                        m1 = ((has2_0 != 0) ? fmaxf(max_1_20, max_1_21) : max_1_20);
                    }
                    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, m0, 2);
                    m0 = fmaxf(m0, _shfl_xor_sync_1);
                    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, m0, 1);
                    m0 = fmaxf(m0, _shfl_xor_sync_2);
                    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, m1, 2);
                    m1 = fmaxf(m1, _shfl_xor_sync_3);
                    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, m1, 1);
                    m1 = fmaxf(m1, _shfl_xor_sync_4);
                    f_max0 = m0;
                    f_max1 = m1;
                    row_max0 = f_max0;
                    row_max1 = f_max1;
                    float bias0 = 8.0f - row_max0;
                    float bias1 = 8.0f - row_max1;
                    float f_sum0 = 0.0f;
                    float f_sum1 = 0.0f;
                    float sa0 = 0.0f;
                    float sa1 = 0.0f;
                    float sb0 = 0.0f;
                    float sb1 = 0.0f;
                    float exp2 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0));
                    float exp2_1 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0));
                    d_qk[0] = exp2;
                    d_qk[1] = exp2_1;
                    sa0 += exp2 + exp2_1;
                    float exp2_2 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[8], bias1));
                    float exp2_3 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[8], bias1));
                    d_qk[2] = exp2_2;
                    d_qk[3] = exp2_3;
                    sa1 += exp2_2 + exp2_3;
                    float exp2_4 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0));
                    float exp2_5 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0));
                    d_qk[4] = exp2_4;
                    d_qk[5] = exp2_5;
                    sa0 += exp2_4 + exp2_5;
                    float exp2_6 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[8], bias1));
                    float exp2_7 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[8], bias1));
                    d_qk[6] = exp2_6;
                    d_qk[7] = exp2_7;
                    sa1 += exp2_6 + exp2_7;
                    float exp2_8 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0));
                    float exp2_9 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0));
                    d_qk[8] = exp2_8;
                    d_qk[9] = exp2_9;
                    sa0 += exp2_8 + exp2_9;
                    float exp2_10 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[9], bias1));
                    float exp2_11 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[9], bias1));
                    d_qk[10] = exp2_10;
                    d_qk[11] = exp2_11;
                    sa1 += exp2_10 + exp2_11;
                    float exp2_12 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0));
                    float exp2_13 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0));
                    d_qk[12] = exp2_12;
                    d_qk[13] = exp2_13;
                    sa0 += exp2_12 + exp2_13;
                    float exp2_14 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[9], bias1));
                    float exp2_15 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[9], bias1));
                    d_qk[14] = exp2_14;
                    d_qk[15] = exp2_15;
                    sa1 += exp2_14 + exp2_15;
                    float exp2_16 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0));
                    float exp2_17 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0));
                    d_qk[16] = exp2_16;
                    d_qk[17] = exp2_17;
                    sa0 += exp2_16 + exp2_17;
                    float exp2_18 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[10], bias1));
                    float exp2_19 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[10], bias1));
                    d_qk[18] = exp2_18;
                    d_qk[19] = exp2_19;
                    sa1 += exp2_18 + exp2_19;
                    float exp2_20 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0));
                    float exp2_21 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0));
                    d_qk[20] = exp2_20;
                    d_qk[21] = exp2_21;
                    sa0 += exp2_20 + exp2_21;
                    float exp2_22 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[10], bias1));
                    float exp2_23 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[10], bias1));
                    d_qk[22] = exp2_22;
                    d_qk[23] = exp2_23;
                    sa1 += exp2_22 + exp2_23;
                    float exp2_24 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0));
                    float exp2_25 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0));
                    d_qk[24] = exp2_24;
                    d_qk[25] = exp2_25;
                    sa0 += exp2_24 + exp2_25;
                    float exp2_26 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[11], bias1));
                    float exp2_27 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[11], bias1));
                    d_qk[26] = exp2_26;
                    d_qk[27] = exp2_27;
                    sa1 += exp2_26 + exp2_27;
                    float exp2_28 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0));
                    float exp2_29 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0));
                    d_qk[28] = exp2_28;
                    d_qk[29] = exp2_29;
                    sa0 += exp2_28 + exp2_29;
                    float exp2_30 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[11], bias1));
                    float exp2_31 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[11], bias1));
                    d_qk[30] = exp2_30;
                    d_qk[31] = exp2_31;
                    sa1 += exp2_30 + exp2_31;
                    float exp2_0 = approx_exp2(__fmaf_rn(d_qk[32], qk_sc[4], bias0));
                    float exp2_1_0 = approx_exp2(__fmaf_rn(d_qk[33], qk_sc[4], bias0));
                    d_qk[32] = exp2_0;
                    d_qk[33] = exp2_1_0;
                    sb0 += exp2_0 + exp2_1_0;
                    float exp2_2_0 = approx_exp2(__fmaf_rn(d_qk[34], qk_sc[12], bias1));
                    float exp2_3_0 = approx_exp2(__fmaf_rn(d_qk[35], qk_sc[12], bias1));
                    d_qk[34] = exp2_2_0;
                    d_qk[35] = exp2_3_0;
                    sb1 += exp2_2_0 + exp2_3_0;
                    float exp2_4_0 = approx_exp2(__fmaf_rn(d_qk[36], qk_sc[4], bias0));
                    float exp2_5_0 = approx_exp2(__fmaf_rn(d_qk[37], qk_sc[4], bias0));
                    d_qk[36] = exp2_4_0;
                    d_qk[37] = exp2_5_0;
                    sb0 += exp2_4_0 + exp2_5_0;
                    float exp2_6_0 = approx_exp2(__fmaf_rn(d_qk[38], qk_sc[12], bias1));
                    float exp2_7_0 = approx_exp2(__fmaf_rn(d_qk[39], qk_sc[12], bias1));
                    d_qk[38] = exp2_6_0;
                    d_qk[39] = exp2_7_0;
                    sb1 += exp2_6_0 + exp2_7_0;
                    float exp2_8_0 = approx_exp2(__fmaf_rn(d_qk[40], qk_sc[5], bias0));
                    float exp2_9_0 = approx_exp2(__fmaf_rn(d_qk[41], qk_sc[5], bias0));
                    d_qk[40] = exp2_8_0;
                    d_qk[41] = exp2_9_0;
                    sb0 += exp2_8_0 + exp2_9_0;
                    float exp2_10_0 = approx_exp2(__fmaf_rn(d_qk[42], qk_sc[13], bias1));
                    float exp2_11_0 = approx_exp2(__fmaf_rn(d_qk[43], qk_sc[13], bias1));
                    d_qk[42] = exp2_10_0;
                    d_qk[43] = exp2_11_0;
                    sb1 += exp2_10_0 + exp2_11_0;
                    float exp2_12_0 = approx_exp2(__fmaf_rn(d_qk[44], qk_sc[5], bias0));
                    float exp2_13_0 = approx_exp2(__fmaf_rn(d_qk[45], qk_sc[5], bias0));
                    d_qk[44] = exp2_12_0;
                    d_qk[45] = exp2_13_0;
                    sb0 += exp2_12_0 + exp2_13_0;
                    float exp2_14_0 = approx_exp2(__fmaf_rn(d_qk[46], qk_sc[13], bias1));
                    float exp2_15_0 = approx_exp2(__fmaf_rn(d_qk[47], qk_sc[13], bias1));
                    d_qk[46] = exp2_14_0;
                    d_qk[47] = exp2_15_0;
                    sb1 += exp2_14_0 + exp2_15_0;
                    float exp2_16_0 = approx_exp2(__fmaf_rn(d_qk[48], qk_sc[6], bias0));
                    float exp2_17_0 = approx_exp2(__fmaf_rn(d_qk[49], qk_sc[6], bias0));
                    d_qk[48] = exp2_16_0;
                    d_qk[49] = exp2_17_0;
                    sb0 += exp2_16_0 + exp2_17_0;
                    float exp2_18_0 = approx_exp2(__fmaf_rn(d_qk[50], qk_sc[14], bias1));
                    float exp2_19_0 = approx_exp2(__fmaf_rn(d_qk[51], qk_sc[14], bias1));
                    d_qk[50] = exp2_18_0;
                    d_qk[51] = exp2_19_0;
                    sb1 += exp2_18_0 + exp2_19_0;
                    float exp2_20_0 = approx_exp2(__fmaf_rn(d_qk[52], qk_sc[6], bias0));
                    float exp2_21_0 = approx_exp2(__fmaf_rn(d_qk[53], qk_sc[6], bias0));
                    d_qk[52] = exp2_20_0;
                    d_qk[53] = exp2_21_0;
                    sb0 += exp2_20_0 + exp2_21_0;
                    float exp2_22_0 = approx_exp2(__fmaf_rn(d_qk[54], qk_sc[14], bias1));
                    float exp2_23_0 = approx_exp2(__fmaf_rn(d_qk[55], qk_sc[14], bias1));
                    d_qk[54] = exp2_22_0;
                    d_qk[55] = exp2_23_0;
                    sb1 += exp2_22_0 + exp2_23_0;
                    float exp2_24_0 = approx_exp2(__fmaf_rn(d_qk[56], qk_sc[7], bias0));
                    float exp2_25_0 = approx_exp2(__fmaf_rn(d_qk[57], qk_sc[7], bias0));
                    d_qk[56] = exp2_24_0;
                    d_qk[57] = exp2_25_0;
                    sb0 += exp2_24_0 + exp2_25_0;
                    float exp2_26_0 = approx_exp2(__fmaf_rn(d_qk[58], qk_sc[15], bias1));
                    float exp2_27_0 = approx_exp2(__fmaf_rn(d_qk[59], qk_sc[15], bias1));
                    d_qk[58] = exp2_26_0;
                    d_qk[59] = exp2_27_0;
                    sb1 += exp2_26_0 + exp2_27_0;
                    float exp2_28_0 = approx_exp2(__fmaf_rn(d_qk[60], qk_sc[7], bias0));
                    float exp2_29_0 = approx_exp2(__fmaf_rn(d_qk[61], qk_sc[7], bias0));
                    d_qk[60] = exp2_28_0;
                    d_qk[61] = exp2_29_0;
                    sb0 += exp2_28_0 + exp2_29_0;
                    float exp2_30_0 = approx_exp2(__fmaf_rn(d_qk[62], qk_sc[15], bias1));
                    float exp2_31_0 = approx_exp2(__fmaf_rn(d_qk[63], qk_sc[15], bias1));
                    d_qk[62] = exp2_30_0;
                    d_qk[63] = exp2_31_0;
                    sb1 += exp2_30_0 + exp2_31_0;
                    f_sum0 += sa0 + ((has2_0 != 0) ? sb0 : 0.0f);
                    f_sum1 += sa1 + ((has2_0 != 0) ? sb1 : 0.0f);
                    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 2);
                    f_sum0 += _shfl_xor_sync_5;
                    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, f_sum0, 1);
                    f_sum0 += _shfl_xor_sync_6;
                    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 2);
                    f_sum1 += _shfl_xor_sync_7;
                    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, f_sum1, 1);
                    f_sum1 += _shfl_xor_sync_8;
                    row_sum0 = f_sum0;
                    row_sum1 = f_sum1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_10;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(d_qk[1]), "f"(d_qk[0]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_11;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(d_qk[5]), "f"(d_qk[4]));
                    uint32_t _prmt_9_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_9_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_10), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_11));
                    p_e4m3[0] = _prmt_9_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_13;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_13) : "f"(d_qk[3]), "f"(d_qk[2]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_14;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(d_qk[7]), "f"(d_qk[6]));
                    uint32_t _prmt_12_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_12_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_13), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_14));
                    p_e4m3[1] = _prmt_12_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_16;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_16) : "f"(d_qk[9]), "f"(d_qk[8]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_17;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(d_qk[13]), "f"(d_qk[12]));
                    uint32_t _prmt_15_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_15_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_16), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_17));
                    p_e4m3[2] = _prmt_15_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_19;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_19) : "f"(d_qk[11]), "f"(d_qk[10]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_20;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_20) : "f"(d_qk[15]), "f"(d_qk[14]));
                    uint32_t _prmt_18_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_18_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_19), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_20));
                    p_e4m3[3] = _prmt_18_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_22;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_22) : "f"(d_qk[17]), "f"(d_qk[16]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_23;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_23) : "f"(d_qk[21]), "f"(d_qk[20]));
                    uint32_t _prmt_21;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_21) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_22), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_23));
                    p_e4m3[4] = _prmt_21;
                    uint16_t _cvt_satfinite_e4m3x2_f32_25;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_25) : "f"(d_qk[19]), "f"(d_qk[18]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_26) : "f"(d_qk[23]), "f"(d_qk[22]));
                    uint32_t _prmt_24_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_24_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_25), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_26));
                    p_e4m3[5] = _prmt_24_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_28;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_28) : "f"(d_qk[25]), "f"(d_qk[24]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_29;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_29) : "f"(d_qk[29]), "f"(d_qk[28]));
                    uint32_t _prmt_27_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_27_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_28), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_29));
                    p_e4m3[6] = _prmt_27_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_31;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_31) : "f"(d_qk[27]), "f"(d_qk[26]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_32) : "f"(d_qk[31]), "f"(d_qk[30]));
                    uint32_t _prmt_30_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_30_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_31), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_32));
                    p_e4m3[7] = _prmt_30_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_34;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_34) : "f"(d_qk[33]), "f"(d_qk[32]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_35;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_35) : "f"(d_qk[37]), "f"(d_qk[36]));
                    uint32_t _prmt_33_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_33_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_34), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_35));
                    p_e4m3[8] = _prmt_33_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_37;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_37) : "f"(d_qk[35]), "f"(d_qk[34]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_38;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_38) : "f"(d_qk[39]), "f"(d_qk[38]));
                    uint32_t _prmt_36;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_36) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_37), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_38));
                    p_e4m3[9] = _prmt_36;
                    uint16_t _cvt_satfinite_e4m3x2_f32_40;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_40) : "f"(d_qk[41]), "f"(d_qk[40]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_41;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_41) : "f"(d_qk[45]), "f"(d_qk[44]));
                    uint32_t _prmt_39_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_39_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_40), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_41));
                    p_e4m3[10] = _prmt_39_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_43;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_43) : "f"(d_qk[43]), "f"(d_qk[42]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_44;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_44) : "f"(d_qk[47]), "f"(d_qk[46]));
                    uint32_t _prmt_42_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_42_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_43), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_44));
                    p_e4m3[11] = _prmt_42_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_46;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_46) : "f"(d_qk[49]), "f"(d_qk[48]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_47;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_47) : "f"(d_qk[53]), "f"(d_qk[52]));
                    uint32_t _prmt_45_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_45_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_46), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_47));
                    p_e4m3[12] = _prmt_45_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_49;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(d_qk[51]), "f"(d_qk[50]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_50;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_50) : "f"(d_qk[55]), "f"(d_qk[54]));
                    uint32_t _prmt_48_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_48_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_49), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_50));
                    p_e4m3[13] = _prmt_48_1;
                    uint16_t _cvt_satfinite_e4m3x2_f32_52;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_52) : "f"(d_qk[57]), "f"(d_qk[56]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_53;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_53) : "f"(d_qk[61]), "f"(d_qk[60]));
                    uint32_t _prmt_51;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_51) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_52), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_53));
                    p_e4m3[14] = _prmt_51;
                    uint16_t _cvt_satfinite_e4m3x2_f32_55;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_55) : "f"(d_qk[59]), "f"(d_qk[58]));
                    uint16_t _cvt_satfinite_e4m3x2_f32_56;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_56) : "f"(d_qk[63]), "f"(d_qk[62]));
                    uint32_t _prmt_54_1;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_54_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_55), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_56));
                    p_e4m3[15] = _prmt_54_1;
                    if (has2_0 == 0) {
                        p_e4m3[8] = 0;
                        p_e4m3[9] = 0;
                        p_e4m3[10] = 0;
                        p_e4m3[11] = 0;
                        p_e4m3[12] = 0;
                        p_e4m3[13] = 0;
                        p_e4m3[14] = 0;
                        p_e4m3[15] = 0;
                    }
                    float pend0 = 1.0f;
                    float pend1 = 1.0f;
                    int pend_need = 0;
                    int n_loop = n_own - 1;
                    #pragma unroll 1
                    for (int n = 0; n < n_loop; n++) {
                        int cur_stage = cur_pos % 3;
                        int w_e_2 = meta_smem[own_base_w + (n + 1 >> 1)];
                        int en = w_e_2 >> (n + 1 & 1) * 16 & 65535;
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
                        int _parity_k_full_3 = (int)((unsigned int)knlap % 2);
                        mbarrier_wait(k_full_addr + (knext % 3) * 8, _parity_k_full_3);
                        int _parity_v_full_2 = (int)((unsigned int)vclap % 2);
                        mbarrier_wait(v_full_addr + (vcur % 3) * 8, _parity_v_full_2);
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_57 = (((uint64_t)(((k_smem_addr + (unsigned int)(nxt_stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_57 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_57);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 0, 1, 1;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]), "l"(wgmma_b_0)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[4]), "r"(q_frag[(4) + 1]), "r"(q_frag[(4) + 2]), "r"(q_frag[(4) + 3]), "l"(wgmma_b_0 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[8]), "r"(q_frag[(8) + 1]), "r"(q_frag[(8) + 2]), "r"(q_frag[(8) + 3]), "l"(wgmma_b_0 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_qk[0]), "+f"(d_qk[1]), "+f"(d_qk[2]), "+f"(d_qk[3]), "+f"(d_qk[4]), "+f"(d_qk[5]), "+f"(d_qk[6]), "+f"(d_qk[7]), "+f"(d_qk[8]), "+f"(d_qk[9]), "+f"(d_qk[10]), "+f"(d_qk[11]), "+f"(d_qk[12]), "+f"(d_qk[13]), "+f"(d_qk[14]), "+f"(d_qk[15]), "+f"(d_qk[16]), "+f"(d_qk[17]), "+f"(d_qk[18]), "+f"(d_qk[19]), "+f"(d_qk[20]), "+f"(d_qk[21]), "+f"(d_qk[22]), "+f"(d_qk[23]), "+f"(d_qk[24]), "+f"(d_qk[25]), "+f"(d_qk[26]), "+f"(d_qk[27]), "+f"(d_qk[28]), "+f"(d_qk[29]), "+f"(d_qk[30]), "+f"(d_qk[31]), "+f"(d_qk[32]), "+f"(d_qk[33]), "+f"(d_qk[34]), "+f"(d_qk[35]), "+f"(d_qk[36]), "+f"(d_qk[37]), "+f"(d_qk[38]), "+f"(d_qk[39]), "+f"(d_qk[40]), "+f"(d_qk[41]), "+f"(d_qk[42]), "+f"(d_qk[43]), "+f"(d_qk[44]), "+f"(d_qk[45]), "+f"(d_qk[46]), "+f"(d_qk[47]), "+f"(d_qk[48]), "+f"(d_qk[49]), "+f"(d_qk[50]), "+f"(d_qk[51]), "+f"(d_qk[52]), "+f"(d_qk[53]), "+f"(d_qk[54]), "+f"(d_qk[55]), "+f"(d_qk[56]), "+f"(d_qk[57]), "+f"(d_qk[58]), "+f"(d_qk[59]), "+f"(d_qk[60]), "+f"(d_qk[61]), "+f"(d_qk[62]), "+f"(d_qk[63])
                            : "r"(q_frag[12]), "r"(q_frag[(12) + 1]), "r"(q_frag[(12) + 2]), "r"(q_frag[(12) + 3]), "l"(wgmma_b_0 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_58 = (((uint64_t)(((vt_smem_addr + (unsigned int)(cur_stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t wgmma_b_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_58 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_58);
                        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(wgmma_b_1), "r"(n + (1 - is_split))
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(wgmma_b_1 + 2)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_e4m3[8]), "r"(p_e4m3[(8) + 1]), "r"(p_e4m3[(8) + 2]), "r"(p_e4m3[(8) + 3]), "l"(wgmma_b_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                            : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                            : "r"(p_e4m3[12]), "r"(p_e4m3[(12) + 1]), "r"(p_e4m3[(12) + 2]), "r"(p_e4m3[(12) + 3]), "l"(wgmma_b_1 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                        float ksg_0 = ks_smem[nxt_stage * 64];
                        qk_sc[0] = q_log2[0] * ksg_0;
                        qk_sc[8] = q_log2[1] * ksg_0;
                        float ksg_1 = ks_smem[nxt_stage * 64 + 1];
                        qk_sc[1] = q_log2[0] * ksg_1;
                        qk_sc[9] = q_log2[1] * ksg_1;
                        float ksg_2 = ks_smem[nxt_stage * 64 + 2];
                        qk_sc[2] = q_log2[0] * ksg_2;
                        qk_sc[10] = q_log2[1] * ksg_2;
                        float ksg_3 = ks_smem[nxt_stage * 64 + 3];
                        qk_sc[3] = q_log2[0] * ksg_3;
                        qk_sc[11] = q_log2[1] * ksg_3;
                        float ksg_4 = ks_smem[nxt_stage * 64 + 32];
                        qk_sc[4] = q_log2[0] * ksg_4;
                        qk_sc[12] = q_log2[1] * ksg_4;
                        float ksg_5 = ks_smem[nxt_stage * 64 + 32 + 1];
                        qk_sc[5] = q_log2[0] * ksg_5;
                        qk_sc[13] = q_log2[1] * ksg_5;
                        float ksg_6 = ks_smem[nxt_stage * 64 + 32 + 2];
                        qk_sc[6] = q_log2[0] * ksg_6;
                        qk_sc[14] = q_log2[1] * ksg_6;
                        float ksg_7 = ks_smem[nxt_stage * 64 + 32 + 3];
                        qk_sc[7] = q_log2[0] * ksg_7;
                        qk_sc[15] = q_log2[1] * ksg_7;
                        if (elect_sync()) {
                            mbarrier_arrive(k_empty_addr + (knext % 3) * 8);
                        }
                        float new_max0 = -CUDART_INF_F;
                        float new_max1 = -CUDART_INF_F;
                        float m0_0 = -CUDART_INF_F;
                        float m1_0 = -CUDART_INF_F;
                        if (scale_log2 >= 0.0f) {
                            float max_1_22 = fmaxf(fmaxf(d_qk[0], d_qk[1]), fmaxf(d_qk[4], d_qk[5]));
                            float max_1_22_1 = fmaxf(fmaxf(d_qk[8], d_qk[9]), fmaxf(d_qk[12], d_qk[13]));
                            float max_1_23 = fmaxf(fmaxf(d_qk[16], d_qk[17]), fmaxf(d_qk[20], d_qk[21]));
                            float max_1_24 = fmaxf(fmaxf(d_qk[24], d_qk[25]), fmaxf(d_qk[28], d_qk[29]));
                            float max_1_25 = fmaxf(fmaxf(d_qk[32], d_qk[33]), fmaxf(d_qk[36], d_qk[37]));
                            float max_1_26 = fmaxf(fmaxf(d_qk[40], d_qk[41]), fmaxf(d_qk[44], d_qk[45]));
                            float max_1_27 = fmaxf(fmaxf(d_qk[48], d_qk[49]), fmaxf(d_qk[52], d_qk[53]));
                            float max_1_28 = fmaxf(fmaxf(d_qk[56], d_qk[57]), fmaxf(d_qk[60], d_qk[61]));
                            float max_1_29 = fmaxf(fmaxf(d_qk[2], d_qk[3]), fmaxf(d_qk[6], d_qk[7]));
                            float max_1_30 = fmaxf(fmaxf(d_qk[10], d_qk[11]), fmaxf(d_qk[14], d_qk[15]));
                            float max_1_31 = fmaxf(fmaxf(d_qk[18], d_qk[19]), fmaxf(d_qk[22], d_qk[23]));
                            float max_1_32 = fmaxf(fmaxf(d_qk[26], d_qk[27]), fmaxf(d_qk[30], d_qk[31]));
                            float max_1_33 = fmaxf(fmaxf(d_qk[34], d_qk[35]), fmaxf(d_qk[38], d_qk[39]));
                            float max_1_34 = fmaxf(fmaxf(d_qk[42], d_qk[43]), fmaxf(d_qk[46], d_qk[47]));
                            float max_1_35 = fmaxf(fmaxf(d_qk[50], d_qk[51]), fmaxf(d_qk[54], d_qk[55]));
                            float max_1_36 = fmaxf(fmaxf(d_qk[58], d_qk[59]), fmaxf(d_qk[62], d_qk[63]));
                            float max_1_37 = fmaxf(fmaxf(max_1_22 * qk_sc[0], max_1_22_1 * qk_sc[1]), fmaxf(max_1_23 * qk_sc[2], max_1_24 * qk_sc[3]));
                            float max_1_38 = fmaxf(fmaxf(max_1_25 * qk_sc[4], max_1_26 * qk_sc[5]), fmaxf(max_1_27 * qk_sc[6], max_1_28 * qk_sc[7]));
                            float max_1_39 = fmaxf(fmaxf(max_1_29 * qk_sc[8], max_1_30 * qk_sc[9]), fmaxf(max_1_31 * qk_sc[10], max_1_32 * qk_sc[11]));
                            float max_1_40 = fmaxf(fmaxf(max_1_33 * qk_sc[12], max_1_34 * qk_sc[13]), fmaxf(max_1_35 * qk_sc[14], max_1_36 * qk_sc[15]));
                            m0_0 = ((nxt_has2 != 0) ? fmaxf(max_1_37, max_1_38) : max_1_37);
                            m1_0 = ((nxt_has2 != 0) ? fmaxf(max_1_39, max_1_40) : max_1_39);
                        } else {
                            float min_1_15 = fminf(fminf(d_qk[0], d_qk[1]), fminf(d_qk[4], d_qk[5]));
                            float min_1_15_1 = fminf(fminf(d_qk[8], d_qk[9]), fminf(d_qk[12], d_qk[13]));
                            float min_1_16 = fminf(fminf(d_qk[16], d_qk[17]), fminf(d_qk[20], d_qk[21]));
                            float min_1_17 = fminf(fminf(d_qk[24], d_qk[25]), fminf(d_qk[28], d_qk[29]));
                            float min_1_18 = fminf(fminf(d_qk[32], d_qk[33]), fminf(d_qk[36], d_qk[37]));
                            float min_1_19 = fminf(fminf(d_qk[40], d_qk[41]), fminf(d_qk[44], d_qk[45]));
                            float min_1_20 = fminf(fminf(d_qk[48], d_qk[49]), fminf(d_qk[52], d_qk[53]));
                            float min_1_21 = fminf(fminf(d_qk[56], d_qk[57]), fminf(d_qk[60], d_qk[61]));
                            float min_1_22 = fminf(fminf(d_qk[2], d_qk[3]), fminf(d_qk[6], d_qk[7]));
                            float min_1_23 = fminf(fminf(d_qk[10], d_qk[11]), fminf(d_qk[14], d_qk[15]));
                            float min_1_24 = fminf(fminf(d_qk[18], d_qk[19]), fminf(d_qk[22], d_qk[23]));
                            float min_1_25 = fminf(fminf(d_qk[26], d_qk[27]), fminf(d_qk[30], d_qk[31]));
                            float min_1_26 = fminf(fminf(d_qk[34], d_qk[35]), fminf(d_qk[38], d_qk[39]));
                            float min_1_27 = fminf(fminf(d_qk[42], d_qk[43]), fminf(d_qk[46], d_qk[47]));
                            float min_1_28 = fminf(fminf(d_qk[50], d_qk[51]), fminf(d_qk[54], d_qk[55]));
                            float min_1_29 = fminf(fminf(d_qk[58], d_qk[59]), fminf(d_qk[62], d_qk[63]));
                            float max_1_41 = fmaxf(fmaxf(min_1_15 * qk_sc[0], min_1_15_1 * qk_sc[1]), fmaxf(min_1_16 * qk_sc[2], min_1_17 * qk_sc[3]));
                            float max_1_41_1 = fmaxf(fmaxf(min_1_18 * qk_sc[4], min_1_19 * qk_sc[5]), fmaxf(min_1_20 * qk_sc[6], min_1_21 * qk_sc[7]));
                            float max_1_42 = fmaxf(fmaxf(min_1_22 * qk_sc[8], min_1_23 * qk_sc[9]), fmaxf(min_1_24 * qk_sc[10], min_1_25 * qk_sc[11]));
                            float max_1_43 = fmaxf(fmaxf(min_1_26 * qk_sc[12], min_1_27 * qk_sc[13]), fmaxf(min_1_28 * qk_sc[14], min_1_29 * qk_sc[15]));
                            m0_0 = ((nxt_has2 != 0) ? fmaxf(max_1_41, max_1_41_1) : max_1_41);
                            m1_0 = ((nxt_has2 != 0) ? fmaxf(max_1_42, max_1_43) : max_1_42);
                        }
                        float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 2);
                        m0_0 = fmaxf(m0_0, _shfl_xor_sync_59);
                        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, m0_0, 1);
                        m0_0 = fmaxf(m0_0, _shfl_xor_sync_60);
                        float _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 2);
                        m1_0 = fmaxf(m1_0, _shfl_xor_sync_61);
                        float _shfl_xor_sync_62 = __shfl_xor_sync(0xFFFFFFFF, m1_0, 1);
                        m1_0 = fmaxf(m1_0, _shfl_xor_sync_62);
                        new_max0 = m0_0;
                        new_max1 = m1_0;
                        float merged_max0 = fmaxf(new_max0, row_max0);
                        float merged_max1 = fmaxf(new_max1, row_max1);
                        int grow0 = merged_max0 - row_max0 > 0.8073549f;
                        int grow1 = merged_max1 - row_max1 > 0.8073549f;
                        float _exp2_63 = approx_exp2(row_max0 - merged_max0);
                        float rescale0 = ((grow0 != 0) ? _exp2_63 : 1.0f);
                        float _exp2_64 = approx_exp2(row_max1 - merged_max1);
                        float rescale1 = ((grow1 != 0) ? _exp2_64 : 1.0f);
                        row_max0 = ((grow0 != 0) ? merged_max0 : row_max0);
                        row_max1 = ((grow1 != 0) ? merged_max1 : row_max1);
                        bias0 = 8.0f - row_max0;
                        bias1 = 8.0f - row_max1;
                        float new_sum0 = 0.0f;
                        float new_sum1 = 0.0f;
                        float sa0_0 = 0.0f;
                        float sa1_0 = 0.0f;
                        float sb0_0 = 0.0f;
                        float sb1_0 = 0.0f;
                        float exp2_32 = approx_exp2(__fmaf_rn(d_qk[0], qk_sc[0], bias0));
                        float exp2_1_1 = approx_exp2(__fmaf_rn(d_qk[1], qk_sc[0], bias0));
                        d_qk[0] = exp2_32;
                        d_qk[1] = exp2_1_1;
                        sa0_0 += exp2_32 + exp2_1_1;
                        float exp2_2_1 = approx_exp2(__fmaf_rn(d_qk[2], qk_sc[8], bias1));
                        float exp2_3_1 = approx_exp2(__fmaf_rn(d_qk[3], qk_sc[8], bias1));
                        d_qk[2] = exp2_2_1;
                        d_qk[3] = exp2_3_1;
                        sa1_0 += exp2_2_1 + exp2_3_1;
                        float exp2_4_1 = approx_exp2(__fmaf_rn(d_qk[4], qk_sc[0], bias0));
                        float exp2_5_1 = approx_exp2(__fmaf_rn(d_qk[5], qk_sc[0], bias0));
                        d_qk[4] = exp2_4_1;
                        d_qk[5] = exp2_5_1;
                        sa0_0 += exp2_4_1 + exp2_5_1;
                        float exp2_6_1 = approx_exp2(__fmaf_rn(d_qk[6], qk_sc[8], bias1));
                        float exp2_7_1 = approx_exp2(__fmaf_rn(d_qk[7], qk_sc[8], bias1));
                        d_qk[6] = exp2_6_1;
                        d_qk[7] = exp2_7_1;
                        sa1_0 += exp2_6_1 + exp2_7_1;
                        float exp2_8_1 = approx_exp2(__fmaf_rn(d_qk[8], qk_sc[1], bias0));
                        float exp2_9_1 = approx_exp2(__fmaf_rn(d_qk[9], qk_sc[1], bias0));
                        d_qk[8] = exp2_8_1;
                        d_qk[9] = exp2_9_1;
                        sa0_0 += exp2_8_1 + exp2_9_1;
                        float exp2_10_1 = approx_exp2(__fmaf_rn(d_qk[10], qk_sc[9], bias1));
                        float exp2_11_1 = approx_exp2(__fmaf_rn(d_qk[11], qk_sc[9], bias1));
                        d_qk[10] = exp2_10_1;
                        d_qk[11] = exp2_11_1;
                        sa1_0 += exp2_10_1 + exp2_11_1;
                        float exp2_12_1 = approx_exp2(__fmaf_rn(d_qk[12], qk_sc[1], bias0));
                        float exp2_13_1 = approx_exp2(__fmaf_rn(d_qk[13], qk_sc[1], bias0));
                        d_qk[12] = exp2_12_1;
                        d_qk[13] = exp2_13_1;
                        sa0_0 += exp2_12_1 + exp2_13_1;
                        float exp2_14_1 = approx_exp2(__fmaf_rn(d_qk[14], qk_sc[9], bias1));
                        float exp2_15_1 = approx_exp2(__fmaf_rn(d_qk[15], qk_sc[9], bias1));
                        d_qk[14] = exp2_14_1;
                        d_qk[15] = exp2_15_1;
                        sa1_0 += exp2_14_1 + exp2_15_1;
                        float exp2_16_1 = approx_exp2(__fmaf_rn(d_qk[16], qk_sc[2], bias0));
                        float exp2_17_1 = approx_exp2(__fmaf_rn(d_qk[17], qk_sc[2], bias0));
                        d_qk[16] = exp2_16_1;
                        d_qk[17] = exp2_17_1;
                        sa0_0 += exp2_16_1 + exp2_17_1;
                        float exp2_18_1 = approx_exp2(__fmaf_rn(d_qk[18], qk_sc[10], bias1));
                        float exp2_19_1 = approx_exp2(__fmaf_rn(d_qk[19], qk_sc[10], bias1));
                        d_qk[18] = exp2_18_1;
                        d_qk[19] = exp2_19_1;
                        sa1_0 += exp2_18_1 + exp2_19_1;
                        float exp2_20_1 = approx_exp2(__fmaf_rn(d_qk[20], qk_sc[2], bias0));
                        float exp2_21_1 = approx_exp2(__fmaf_rn(d_qk[21], qk_sc[2], bias0));
                        d_qk[20] = exp2_20_1;
                        d_qk[21] = exp2_21_1;
                        sa0_0 += exp2_20_1 + exp2_21_1;
                        float exp2_22_1 = approx_exp2(__fmaf_rn(d_qk[22], qk_sc[10], bias1));
                        float exp2_23_1 = approx_exp2(__fmaf_rn(d_qk[23], qk_sc[10], bias1));
                        d_qk[22] = exp2_22_1;
                        d_qk[23] = exp2_23_1;
                        sa1_0 += exp2_22_1 + exp2_23_1;
                        float exp2_24_1 = approx_exp2(__fmaf_rn(d_qk[24], qk_sc[3], bias0));
                        float exp2_25_1 = approx_exp2(__fmaf_rn(d_qk[25], qk_sc[3], bias0));
                        d_qk[24] = exp2_24_1;
                        d_qk[25] = exp2_25_1;
                        sa0_0 += exp2_24_1 + exp2_25_1;
                        float exp2_26_1 = approx_exp2(__fmaf_rn(d_qk[26], qk_sc[11], bias1));
                        float exp2_27_1 = approx_exp2(__fmaf_rn(d_qk[27], qk_sc[11], bias1));
                        d_qk[26] = exp2_26_1;
                        d_qk[27] = exp2_27_1;
                        sa1_0 += exp2_26_1 + exp2_27_1;
                        float exp2_28_1 = approx_exp2(__fmaf_rn(d_qk[28], qk_sc[3], bias0));
                        float exp2_29_1 = approx_exp2(__fmaf_rn(d_qk[29], qk_sc[3], bias0));
                        d_qk[28] = exp2_28_1;
                        d_qk[29] = exp2_29_1;
                        sa0_0 += exp2_28_1 + exp2_29_1;
                        float exp2_30_1 = approx_exp2(__fmaf_rn(d_qk[30], qk_sc[11], bias1));
                        float exp2_31_1 = approx_exp2(__fmaf_rn(d_qk[31], qk_sc[11], bias1));
                        d_qk[30] = exp2_30_1;
                        d_qk[31] = exp2_31_1;
                        sa1_0 += exp2_30_1 + exp2_31_1;
                        float exp2_33 = approx_exp2(__fmaf_rn(d_qk[32], qk_sc[4], bias0));
                        float exp2_1_2 = approx_exp2(__fmaf_rn(d_qk[33], qk_sc[4], bias0));
                        d_qk[32] = exp2_33;
                        d_qk[33] = exp2_1_2;
                        sb0_0 += exp2_33 + exp2_1_2;
                        float exp2_2_2 = approx_exp2(__fmaf_rn(d_qk[34], qk_sc[12], bias1));
                        float exp2_3_2 = approx_exp2(__fmaf_rn(d_qk[35], qk_sc[12], bias1));
                        d_qk[34] = exp2_2_2;
                        d_qk[35] = exp2_3_2;
                        sb1_0 += exp2_2_2 + exp2_3_2;
                        float exp2_4_2 = approx_exp2(__fmaf_rn(d_qk[36], qk_sc[4], bias0));
                        float exp2_5_2 = approx_exp2(__fmaf_rn(d_qk[37], qk_sc[4], bias0));
                        d_qk[36] = exp2_4_2;
                        d_qk[37] = exp2_5_2;
                        sb0_0 += exp2_4_2 + exp2_5_2;
                        float exp2_6_2 = approx_exp2(__fmaf_rn(d_qk[38], qk_sc[12], bias1));
                        float exp2_7_2 = approx_exp2(__fmaf_rn(d_qk[39], qk_sc[12], bias1));
                        d_qk[38] = exp2_6_2;
                        d_qk[39] = exp2_7_2;
                        sb1_0 += exp2_6_2 + exp2_7_2;
                        float exp2_8_2 = approx_exp2(__fmaf_rn(d_qk[40], qk_sc[5], bias0));
                        float exp2_9_2 = approx_exp2(__fmaf_rn(d_qk[41], qk_sc[5], bias0));
                        d_qk[40] = exp2_8_2;
                        d_qk[41] = exp2_9_2;
                        sb0_0 += exp2_8_2 + exp2_9_2;
                        float exp2_10_2 = approx_exp2(__fmaf_rn(d_qk[42], qk_sc[13], bias1));
                        float exp2_11_2 = approx_exp2(__fmaf_rn(d_qk[43], qk_sc[13], bias1));
                        d_qk[42] = exp2_10_2;
                        d_qk[43] = exp2_11_2;
                        sb1_0 += exp2_10_2 + exp2_11_2;
                        float exp2_12_2 = approx_exp2(__fmaf_rn(d_qk[44], qk_sc[5], bias0));
                        float exp2_13_2 = approx_exp2(__fmaf_rn(d_qk[45], qk_sc[5], bias0));
                        d_qk[44] = exp2_12_2;
                        d_qk[45] = exp2_13_2;
                        sb0_0 += exp2_12_2 + exp2_13_2;
                        float exp2_14_2 = approx_exp2(__fmaf_rn(d_qk[46], qk_sc[13], bias1));
                        float exp2_15_2 = approx_exp2(__fmaf_rn(d_qk[47], qk_sc[13], bias1));
                        d_qk[46] = exp2_14_2;
                        d_qk[47] = exp2_15_2;
                        sb1_0 += exp2_14_2 + exp2_15_2;
                        float exp2_16_2 = approx_exp2(__fmaf_rn(d_qk[48], qk_sc[6], bias0));
                        float exp2_17_2 = approx_exp2(__fmaf_rn(d_qk[49], qk_sc[6], bias0));
                        d_qk[48] = exp2_16_2;
                        d_qk[49] = exp2_17_2;
                        sb0_0 += exp2_16_2 + exp2_17_2;
                        float exp2_18_2 = approx_exp2(__fmaf_rn(d_qk[50], qk_sc[14], bias1));
                        float exp2_19_2 = approx_exp2(__fmaf_rn(d_qk[51], qk_sc[14], bias1));
                        d_qk[50] = exp2_18_2;
                        d_qk[51] = exp2_19_2;
                        sb1_0 += exp2_18_2 + exp2_19_2;
                        float exp2_20_2 = approx_exp2(__fmaf_rn(d_qk[52], qk_sc[6], bias0));
                        float exp2_21_2 = approx_exp2(__fmaf_rn(d_qk[53], qk_sc[6], bias0));
                        d_qk[52] = exp2_20_2;
                        d_qk[53] = exp2_21_2;
                        sb0_0 += exp2_20_2 + exp2_21_2;
                        float exp2_22_2 = approx_exp2(__fmaf_rn(d_qk[54], qk_sc[14], bias1));
                        float exp2_23_2 = approx_exp2(__fmaf_rn(d_qk[55], qk_sc[14], bias1));
                        d_qk[54] = exp2_22_2;
                        d_qk[55] = exp2_23_2;
                        sb1_0 += exp2_22_2 + exp2_23_2;
                        float exp2_24_2 = approx_exp2(__fmaf_rn(d_qk[56], qk_sc[7], bias0));
                        float exp2_25_2 = approx_exp2(__fmaf_rn(d_qk[57], qk_sc[7], bias0));
                        d_qk[56] = exp2_24_2;
                        d_qk[57] = exp2_25_2;
                        sb0_0 += exp2_24_2 + exp2_25_2;
                        float exp2_26_2 = approx_exp2(__fmaf_rn(d_qk[58], qk_sc[15], bias1));
                        float exp2_27_2 = approx_exp2(__fmaf_rn(d_qk[59], qk_sc[15], bias1));
                        d_qk[58] = exp2_26_2;
                        d_qk[59] = exp2_27_2;
                        sb1_0 += exp2_26_2 + exp2_27_2;
                        float exp2_28_2 = approx_exp2(__fmaf_rn(d_qk[60], qk_sc[7], bias0));
                        float exp2_29_2 = approx_exp2(__fmaf_rn(d_qk[61], qk_sc[7], bias0));
                        d_qk[60] = exp2_28_2;
                        d_qk[61] = exp2_29_2;
                        sb0_0 += exp2_28_2 + exp2_29_2;
                        float exp2_30_2 = approx_exp2(__fmaf_rn(d_qk[62], qk_sc[15], bias1));
                        float exp2_31_2 = approx_exp2(__fmaf_rn(d_qk[63], qk_sc[15], bias1));
                        d_qk[62] = exp2_30_2;
                        d_qk[63] = exp2_31_2;
                        sb1_0 += exp2_30_2 + exp2_31_2;
                        new_sum0 += sa0_0 + ((nxt_has2 != 0) ? sb0_0 : 0.0f);
                        new_sum1 += sa1_0 + ((nxt_has2 != 0) ? sb1_0 : 0.0f);
                        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 2);
                        new_sum0 += _shfl_xor_sync_65;
                        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, new_sum0, 1);
                        new_sum0 += _shfl_xor_sync_66;
                        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 2);
                        new_sum1 += _shfl_xor_sync_67;
                        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, new_sum1, 1);
                        new_sum1 += _shfl_xor_sync_68;
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(v_empty_addr + (vcur % 3) * 8);
                        }
                        pend0 = rescale0;
                        pend1 = rescale1;
                        int _warp_vote_69 = __any_sync(0xFFFFFFFF, rescale0 != 1.0f || rescale1 != 1.0f);
                        pend_need = _warp_vote_69;
                        row_sum0 = __fmaf_rn(row_sum0, rescale0, new_sum0);
                        row_sum1 = __fmaf_rn(row_sum1, rescale1, new_sum1);
                        uint16_t _cvt_satfinite_e4m3x2_f32_71;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_71) : "f"(d_qk[1]), "f"(d_qk[0]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_72;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_72) : "f"(d_qk[5]), "f"(d_qk[4]));
                        uint32_t _prmt_70_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_70_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_71), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_72));
                        p_e4m3[0] = _prmt_70_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_74;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_74) : "f"(d_qk[3]), "f"(d_qk[2]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_75;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_75) : "f"(d_qk[7]), "f"(d_qk[6]));
                        uint32_t _prmt_73_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_73_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_74), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_75));
                        p_e4m3[1] = _prmt_73_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_77;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_77) : "f"(d_qk[9]), "f"(d_qk[8]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_78;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_78) : "f"(d_qk[13]), "f"(d_qk[12]));
                        uint32_t _prmt_76;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_76) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_77), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_78));
                        p_e4m3[2] = _prmt_76;
                        uint16_t _cvt_satfinite_e4m3x2_f32_80;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_80) : "f"(d_qk[11]), "f"(d_qk[10]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_81;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_81) : "f"(d_qk[15]), "f"(d_qk[14]));
                        uint32_t _prmt_79_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_79_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_80), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_81));
                        p_e4m3[3] = _prmt_79_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_83;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_83) : "f"(d_qk[17]), "f"(d_qk[16]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_84;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_84) : "f"(d_qk[21]), "f"(d_qk[20]));
                        uint32_t _prmt_82_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_82_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_83), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_84));
                        p_e4m3[4] = _prmt_82_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_86;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_86) : "f"(d_qk[19]), "f"(d_qk[18]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_87;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_87) : "f"(d_qk[23]), "f"(d_qk[22]));
                        uint32_t _prmt_85_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_85_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_86), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_87));
                        p_e4m3[5] = _prmt_85_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_89;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_89) : "f"(d_qk[25]), "f"(d_qk[24]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_90;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_90) : "f"(d_qk[29]), "f"(d_qk[28]));
                        uint32_t _prmt_88_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_88_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_89), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_90));
                        p_e4m3[6] = _prmt_88_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_92;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_92) : "f"(d_qk[27]), "f"(d_qk[26]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_93;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_93) : "f"(d_qk[31]), "f"(d_qk[30]));
                        uint32_t _prmt_91;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_91) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_92), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_93));
                        p_e4m3[7] = _prmt_91;
                        uint16_t _cvt_satfinite_e4m3x2_f32_95;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_95) : "f"(d_qk[33]), "f"(d_qk[32]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_96;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(d_qk[37]), "f"(d_qk[36]));
                        uint32_t _prmt_94_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_94_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_95), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_96));
                        p_e4m3[8] = _prmt_94_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_98;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_98) : "f"(d_qk[35]), "f"(d_qk[34]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_99;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_99) : "f"(d_qk[39]), "f"(d_qk[38]));
                        uint32_t _prmt_97_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_97_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_98), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_99));
                        p_e4m3[9] = _prmt_97_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_101;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_101) : "f"(d_qk[41]), "f"(d_qk[40]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_102;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_102) : "f"(d_qk[45]), "f"(d_qk[44]));
                        uint32_t _prmt_100_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_100_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_101), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_102));
                        p_e4m3[10] = _prmt_100_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_104;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_104) : "f"(d_qk[43]), "f"(d_qk[42]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_105;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_105) : "f"(d_qk[47]), "f"(d_qk[46]));
                        uint32_t _prmt_103_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_103_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_104), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_105));
                        p_e4m3[11] = _prmt_103_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_107;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_107) : "f"(d_qk[49]), "f"(d_qk[48]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_108;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_108) : "f"(d_qk[53]), "f"(d_qk[52]));
                        uint32_t _prmt_106;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_106) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_107), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_108));
                        p_e4m3[12] = _prmt_106;
                        uint16_t _cvt_satfinite_e4m3x2_f32_110;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_110) : "f"(d_qk[51]), "f"(d_qk[50]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_111;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_111) : "f"(d_qk[55]), "f"(d_qk[54]));
                        uint32_t _prmt_109_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_109_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_110), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_111));
                        p_e4m3[13] = _prmt_109_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_113;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_113) : "f"(d_qk[57]), "f"(d_qk[56]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_114;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_114) : "f"(d_qk[61]), "f"(d_qk[60]));
                        uint32_t _prmt_112_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_112_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_113), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_114));
                        p_e4m3[14] = _prmt_112_1;
                        uint16_t _cvt_satfinite_e4m3x2_f32_116;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_116) : "f"(d_qk[59]), "f"(d_qk[58]));
                        uint16_t _cvt_satfinite_e4m3x2_f32_117;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_117) : "f"(d_qk[63]), "f"(d_qk[62]));
                        uint32_t _prmt_115_1;
                        asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_115_1) : "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_116), "r"((unsigned int)_cvt_satfinite_e4m3x2_f32_117));
                        p_e4m3[15] = _prmt_115_1;
                        if (nxt_has2 == 0) {
                            p_e4m3[8] = 0;
                            p_e4m3[9] = 0;
                            p_e4m3[10] = 0;
                            p_e4m3[11] = 0;
                            p_e4m3[12] = 0;
                            p_e4m3[13] = 0;
                            p_e4m3[14] = 0;
                            p_e4m3[15] = 0;
                        }
                        prev = nxt_pos;
                        cur_pos = nxt_pos;
                        cur_has2 = nxt_has2;
                    }
                    int last_stage = cur_pos % 3;
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
                    int vlast = 0;
                    vlast += cur_pos;
                    int vllap = 0;
                    vllap += cur_pos / 3;
                    int _parity_v_full_3 = (int)((unsigned int)vllap % 2);
                    mbarrier_wait(v_full_addr + (vlast % 3) * 8, _parity_v_full_3);
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_118 = (((uint64_t)(((vt_smem_addr + (unsigned int)(last_stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t wgmma_b_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_118 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_118);
                    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_e4m3[0]), "r"(p_e4m3[1]), "r"(p_e4m3[2]), "r"(p_e4m3[3]), "l"(wgmma_b_2), "r"(n_own - 1 + (1 - is_split))
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_e4m3[4]), "r"(p_e4m3[(4) + 1]), "r"(p_e4m3[(4) + 2]), "r"(p_e4m3[(4) + 3]), "l"(wgmma_b_2 + 2)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_e4m3[8]), "r"(p_e4m3[(8) + 1]), "r"(p_e4m3[(8) + 2]), "r"(p_e4m3[(8) + 3]), "l"(wgmma_b_2 + 4)
                        : "memory");
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(p_e4m3[12]), "r"(p_e4m3[(12) + 1]), "r"(p_e4m3[(12) + 2]), "r"(p_e4m3[(12) + 3]), "l"(wgmma_b_2 + 6)
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
                    int tlap_1 = 0;
                    tlap_1 += p_3 / 3;
                    int _parity_k_full_4 = (int)((unsigned int)tlap_1 % 2);
                    mbarrier_wait(k_full_addr + (p_3 % 3) * 8, _parity_k_full_4);
                    if (elect_sync()) {
                        mbarrier_arrive(k_empty_addr + (p_3 % 3) * 8);
                    }
                    int _parity_v_full_4 = (int)((unsigned int)tlap_1 % 2);
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
                        float _exp2_119 = approx_exp2(row_max0 - mm0);
                        a0 = ((row_max0 == -CUDART_INF_F) ? 0.0f : _exp2_119);
                        float _exp2_120 = approx_exp2(row_max1 - mm1);
                        a1 = ((row_max1 == -CUDART_INF_F) ? 0.0f : _exp2_120);
                        float _exp2_121 = approx_exp2(pm0 - mm0);
                        b0 = ((pm0 == -CUDART_INF_F) ? 0.0f : _exp2_121);
                        float _exp2_122 = approx_exp2(pm1 - mm1);
                        b1 = ((pm1 == -CUDART_INF_F) ? 0.0f : _exp2_122);
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
                    int vsb = mbase + 144 + 2 * qt;
                    float vs0_1_0 = vs_smem[vsb];
                    float vs1_2_0 = vs_smem[vsb + 1];
                    d_o[0] = d_o[0] * (vs0_1_0 * _rcp_0);
                    d_o[1] = d_o[1] * (vs1_2_0 * _rcp_0);
                    d_o[2] = d_o[2] * (vs0_1_0 * _rcp_1);
                    d_o[3] = d_o[3] * (vs1_2_0 * _rcp_1);
                    float vs0_1_1 = vs_smem[vsb + 8];
                    float vs1_2_1 = vs_smem[vsb + 8 + 1];
                    d_o[4] = d_o[4] * (vs0_1_1 * _rcp_0);
                    d_o[5] = d_o[5] * (vs1_2_1 * _rcp_0);
                    d_o[6] = d_o[6] * (vs0_1_1 * _rcp_1);
                    d_o[7] = d_o[7] * (vs1_2_1 * _rcp_1);
                    float vs0_1_2 = vs_smem[vsb + 16];
                    float vs1_2_2 = vs_smem[vsb + 16 + 1];
                    d_o[8] = d_o[8] * (vs0_1_2 * _rcp_0);
                    d_o[9] = d_o[9] * (vs1_2_2 * _rcp_0);
                    d_o[10] = d_o[10] * (vs0_1_2 * _rcp_1);
                    d_o[11] = d_o[11] * (vs1_2_2 * _rcp_1);
                    float vs0_1_3 = vs_smem[vsb + 24];
                    float vs1_2_3 = vs_smem[vsb + 24 + 1];
                    d_o[12] = d_o[12] * (vs0_1_3 * _rcp_0);
                    d_o[13] = d_o[13] * (vs1_2_3 * _rcp_0);
                    d_o[14] = d_o[14] * (vs0_1_3 * _rcp_1);
                    d_o[15] = d_o[15] * (vs1_2_3 * _rcp_1);
                    float vs0_1_4 = vs_smem[vsb + 32];
                    float vs1_2_4 = vs_smem[vsb + 32 + 1];
                    d_o[16] = d_o[16] * (vs0_1_4 * _rcp_0);
                    d_o[17] = d_o[17] * (vs1_2_4 * _rcp_0);
                    d_o[18] = d_o[18] * (vs0_1_4 * _rcp_1);
                    d_o[19] = d_o[19] * (vs1_2_4 * _rcp_1);
                    float vs0_1_5 = vs_smem[vsb + 40];
                    float vs1_2_5 = vs_smem[vsb + 40 + 1];
                    d_o[20] = d_o[20] * (vs0_1_5 * _rcp_0);
                    d_o[21] = d_o[21] * (vs1_2_5 * _rcp_0);
                    d_o[22] = d_o[22] * (vs0_1_5 * _rcp_1);
                    d_o[23] = d_o[23] * (vs1_2_5 * _rcp_1);
                    float vs0_1_6 = vs_smem[vsb + 48];
                    float vs1_2_6 = vs_smem[vsb + 48 + 1];
                    d_o[24] = d_o[24] * (vs0_1_6 * _rcp_0);
                    d_o[25] = d_o[25] * (vs1_2_6 * _rcp_0);
                    d_o[26] = d_o[26] * (vs0_1_6 * _rcp_1);
                    d_o[27] = d_o[27] * (vs1_2_6 * _rcp_1);
                    float vs0_1_7 = vs_smem[vsb + 56];
                    float vs1_2_7 = vs_smem[vsb + 56 + 1];
                    d_o[28] = d_o[28] * (vs0_1_7 * _rcp_0);
                    d_o[29] = d_o[29] * (vs1_2_7 * _rcp_0);
                    d_o[30] = d_o[30] * (vs0_1_7 * _rcp_1);
                    d_o[31] = d_o[31] * (vs1_2_7 * _rcp_1);
                    float vs0_1_8 = vs_smem[vsb + 64];
                    float vs1_2_8 = vs_smem[vsb + 64 + 1];
                    d_o[32] = d_o[32] * (vs0_1_8 * _rcp_0);
                    d_o[33] = d_o[33] * (vs1_2_8 * _rcp_0);
                    d_o[34] = d_o[34] * (vs0_1_8 * _rcp_1);
                    d_o[35] = d_o[35] * (vs1_2_8 * _rcp_1);
                    float vs0_1_9 = vs_smem[vsb + 72];
                    float vs1_2_9 = vs_smem[vsb + 72 + 1];
                    d_o[36] = d_o[36] * (vs0_1_9 * _rcp_0);
                    d_o[37] = d_o[37] * (vs1_2_9 * _rcp_0);
                    d_o[38] = d_o[38] * (vs0_1_9 * _rcp_1);
                    d_o[39] = d_o[39] * (vs1_2_9 * _rcp_1);
                    float vs0_1_10 = vs_smem[vsb + 80];
                    float vs1_2_10 = vs_smem[vsb + 80 + 1];
                    d_o[40] = d_o[40] * (vs0_1_10 * _rcp_0);
                    d_o[41] = d_o[41] * (vs1_2_10 * _rcp_0);
                    d_o[42] = d_o[42] * (vs0_1_10 * _rcp_1);
                    d_o[43] = d_o[43] * (vs1_2_10 * _rcp_1);
                    float vs0_1_11 = vs_smem[vsb + 88];
                    float vs1_2_11 = vs_smem[vsb + 88 + 1];
                    d_o[44] = d_o[44] * (vs0_1_11 * _rcp_0);
                    d_o[45] = d_o[45] * (vs1_2_11 * _rcp_0);
                    d_o[46] = d_o[46] * (vs0_1_11 * _rcp_1);
                    d_o[47] = d_o[47] * (vs1_2_11 * _rcp_1);
                    float vs0_1_12 = vs_smem[vsb + 96];
                    float vs1_2_12 = vs_smem[vsb + 96 + 1];
                    d_o[48] = d_o[48] * (vs0_1_12 * _rcp_0);
                    d_o[49] = d_o[49] * (vs1_2_12 * _rcp_0);
                    d_o[50] = d_o[50] * (vs0_1_12 * _rcp_1);
                    d_o[51] = d_o[51] * (vs1_2_12 * _rcp_1);
                    float vs0_1_13 = vs_smem[vsb + 104];
                    float vs1_2_13 = vs_smem[vsb + 104 + 1];
                    d_o[52] = d_o[52] * (vs0_1_13 * _rcp_0);
                    d_o[53] = d_o[53] * (vs1_2_13 * _rcp_0);
                    d_o[54] = d_o[54] * (vs0_1_13 * _rcp_1);
                    d_o[55] = d_o[55] * (vs1_2_13 * _rcp_1);
                    float vs0_1_14 = vs_smem[vsb + 112];
                    float vs1_2_14 = vs_smem[vsb + 112 + 1];
                    d_o[56] = d_o[56] * (vs0_1_14 * _rcp_0);
                    d_o[57] = d_o[57] * (vs1_2_14 * _rcp_0);
                    d_o[58] = d_o[58] * (vs0_1_14 * _rcp_1);
                    d_o[59] = d_o[59] * (vs1_2_14 * _rcp_1);
                    float vs0_1_15 = vs_smem[vsb + 120];
                    float vs1_2_15 = vs_smem[vsb + 120 + 1];
                    d_o[60] = d_o[60] * (vs0_1_15 * _rcp_0);
                    d_o[61] = d_o[61] * (vs1_2_15 * _rcp_0);
                    d_o[62] = d_o[62] * (vs0_1_15 * _rcp_1);
                    d_o[63] = d_o[63] * (vs1_2_15 * _rcp_1);
                    int qj = lane & 3;
                    int qodd = qj & 1;
                    unsigned int o_pk[2];
                    unsigned int o_vec[2];
                    int o_row_base = (head_o * seqlen_q + my_qb * 64) * 128 + qj * 2 + qodd * 6;
                    int m_local_r_1_0 = m0_local;
                    int row_off_1_0 = o_row_base + m_local_r_1_0 * 128;
                    __nv_bfloat162 _bit_cast_src_123 = __float22bfloat162_rn(make_float2(d_o[0], d_o[1]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_123)[0];
                    __nv_bfloat162 _bit_cast_src_124 = __float22bfloat162_rn(make_float2(d_o[4], d_o[5]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_124)[0];
                    unsigned int send_1_0 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send_1_0, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_0);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_0 : o_pk[1]);
                    reinterpret_cast<int2*>(O + row_off_1_0)[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_125 = __float22bfloat162_rn(make_float2(d_o[8], d_o[9]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_125)[0];
                    __nv_bfloat162 _bit_cast_src_126 = __float22bfloat162_rn(make_float2(d_o[12], d_o[13]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_126)[0];
                    unsigned int send_1_1 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send_1_1, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_1);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_1 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_127 = __float22bfloat162_rn(make_float2(d_o[16], d_o[17]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_127)[0];
                    __nv_bfloat162 _bit_cast_src_128 = __float22bfloat162_rn(make_float2(d_o[20], d_o[21]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_128)[0];
                    unsigned int send_1_2 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send_1_2, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_2);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_2 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_129 = __float22bfloat162_rn(make_float2(d_o[24], d_o[25]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_129)[0];
                    __nv_bfloat162 _bit_cast_src_130 = __float22bfloat162_rn(make_float2(d_o[28], d_o[29]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_130)[0];
                    unsigned int send_1_3 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send_1_3, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_3);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_3 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_131 = __float22bfloat162_rn(make_float2(d_o[32], d_o[33]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_131)[0];
                    __nv_bfloat162 _bit_cast_src_132 = __float22bfloat162_rn(make_float2(d_o[36], d_o[37]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_132)[0];
                    unsigned int send_1_4 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, send_1_4, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_4);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_4 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 64))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_133 = __float22bfloat162_rn(make_float2(d_o[40], d_o[41]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_133)[0];
                    __nv_bfloat162 _bit_cast_src_134 = __float22bfloat162_rn(make_float2(d_o[44], d_o[45]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_134)[0];
                    unsigned int send_1_5 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send_1_5, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_5);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_5 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 80))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_135 = __float22bfloat162_rn(make_float2(d_o[48], d_o[49]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_135)[0];
                    __nv_bfloat162 _bit_cast_src_136 = __float22bfloat162_rn(make_float2(d_o[52], d_o[53]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_136)[0];
                    unsigned int send_1_6 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send_1_6, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_6);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_6 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 96))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_137 = __float22bfloat162_rn(make_float2(d_o[56], d_o[57]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_137)[0];
                    __nv_bfloat162 _bit_cast_src_138 = __float22bfloat162_rn(make_float2(d_o[60], d_o[61]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_138)[0];
                    unsigned int send_1_7 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send_1_7, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_7);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_7 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_0 + 112))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    int m_local_r_1_1 = m1_local;
                    int row_off_1_1 = o_row_base + m_local_r_1_1 * 128;
                    __nv_bfloat162 _bit_cast_src_139 = __float22bfloat162_rn(make_float2(d_o[2], d_o[3]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_139)[0];
                    __nv_bfloat162 _bit_cast_src_140 = __float22bfloat162_rn(make_float2(d_o[6], d_o[7]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_140)[0];
                    unsigned int send_1_8 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send_1_8, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_8);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_8 : o_pk[1]);
                    reinterpret_cast<int2*>(O + row_off_1_1)[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_141 = __float22bfloat162_rn(make_float2(d_o[10], d_o[11]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_141)[0];
                    __nv_bfloat162 _bit_cast_src_142 = __float22bfloat162_rn(make_float2(d_o[14], d_o[15]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_142)[0];
                    unsigned int send_1_9 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, send_1_9, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_9);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_9 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 16))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_143 = __float22bfloat162_rn(make_float2(d_o[18], d_o[19]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_143)[0];
                    __nv_bfloat162 _bit_cast_src_144 = __float22bfloat162_rn(make_float2(d_o[22], d_o[23]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_144)[0];
                    unsigned int send_1_10 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send_1_10, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_10);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_10 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 32))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_145 = __float22bfloat162_rn(make_float2(d_o[26], d_o[27]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_145)[0];
                    __nv_bfloat162 _bit_cast_src_146 = __float22bfloat162_rn(make_float2(d_o[30], d_o[31]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_146)[0];
                    unsigned int send_1_11 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send_1_11, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_11);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_11 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 48))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_147 = __float22bfloat162_rn(make_float2(d_o[34], d_o[35]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_147)[0];
                    __nv_bfloat162 _bit_cast_src_148 = __float22bfloat162_rn(make_float2(d_o[38], d_o[39]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_148)[0];
                    unsigned int send_1_12 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send_1_12, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_12);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_12 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 64))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_149 = __float22bfloat162_rn(make_float2(d_o[42], d_o[43]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_149)[0];
                    __nv_bfloat162 _bit_cast_src_150 = __float22bfloat162_rn(make_float2(d_o[46], d_o[47]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_150)[0];
                    unsigned int send_1_13 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send_1_13, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_13);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_13 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 80))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_151 = __float22bfloat162_rn(make_float2(d_o[50], d_o[51]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_151)[0];
                    __nv_bfloat162 _bit_cast_src_152 = __float22bfloat162_rn(make_float2(d_o[54], d_o[55]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_152)[0];
                    unsigned int send_1_14 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, send_1_14, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_14);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_14 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 96))[0] = reinterpret_cast<int2*>(o_vec)[0];
                    __nv_bfloat162 _bit_cast_src_153 = __float22bfloat162_rn(make_float2(d_o[58], d_o[59]));
                    o_pk[0] = reinterpret_cast<unsigned int*>(&_bit_cast_src_153)[0];
                    __nv_bfloat162 _bit_cast_src_154 = __float22bfloat162_rn(make_float2(d_o[62], d_o[63]));
                    o_pk[1] = reinterpret_cast<unsigned int*>(&_bit_cast_src_154)[0];
                    unsigned int send_1_15 = ((qodd != 0) ? o_pk[0] : o_pk[1]);
                    unsigned int _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send_1_15, 1);
                    o_vec[0] = ((qodd == 0) ? o_pk[0] : _shfl_xor_15);
                    o_vec[1] = ((qodd == 0) ? _shfl_xor_15 : o_pk[1]);
                    reinterpret_cast<int2*>(O + (row_off_1_1 + 112))[0] = reinterpret_cast<int2*>(o_vec)[0];
                }
                mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)ti_4 % 2)) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
